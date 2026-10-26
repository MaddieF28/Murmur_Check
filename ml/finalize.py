"""Turns an experiment into the app's model: average the 5 folds, export ONNX, pick cutoff, score test kids once.
Usage: .venv/bin/python ml/finalize.py NAME [key=value ...] [--install] [--target 0.85]
"""
import json
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch

import experiments as E
from crossval import Ensemble
from data import load_circor, load_labels
from model import SR
from train import auroc, single_file

ROOT = E.ROOT
SPOTS = ("PV", "TV")  # top + bottom of breastbone


def child_scores(recs, scores, cond):
    kids = {}
    for r in recs:
        if r["spot"] in SPOTS:
            k = kids.setdefault(r["pid"], dict(y=r["child_murmur"], spots=set(), s=-np.inf))
            k["spots"].add(r["spot"])
            k["s"] = max(k["s"], scores[r["stem"]][cond])
    kids = [k for k in kids.values() if k["spots"] >= set(SPOTS)]
    return np.array([k["y"] for k in kids]), np.array([k["s"] for k in kids])


def at_cutoff(y, s, cutoff):
    return dict(children=int(len(y)), murmur_children=int(y.sum()), auroc=round(auroc(y, s), 4),
                sensitivity=round(float((s[y] >= cutoff).mean()), 4),
                specificity=round(float((s[~y] < cutoff).mean()), 4))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name, overrides = args[0], args[1:]
    install = "--install" in sys.argv
    target = float(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else 0.85
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # NAME can be a few experiments joined with "+"
    folders = []
    for n in name.split("+"):
        folder = E.EXP / n if (E.EXP / n).exists() else ROOT / "ml/out/exp_pc" / n
        if not (folder / "scores.json").exists() or len(list(folder.glob("fold*.pt"))) < 5:
            E.run(n, E.parse(overrides + ["folds=0,1,2,3,4"]))
            folder = E.EXP / n
        folders.append(folder)
    cfgs = [dict(E.DEFAULTS, **json.load(open(f / "config.json"))) for f in folders]
    assert len({c["win_s"] for c in cfgs}) == 1, "all models in an ensemble need the same window length"
    cfg, win = cfgs[0], int(cfgs[0]["win_s"] * SR)

    nets = []
    for f, c in zip(folders, cfgs):
        for k in range(5):
            m = E.build_model(c)
            m.load_state_dict(torch.load(f / f"fold{k}.pt", map_location="cpu"))
            nets.append(m.eval())
    ens = Ensemble(nets).eval()
    x = torch.randn(3, win) * 0.1
    folder = E.EXP / f"final_{name.replace('+', '__')}"
    folder.mkdir(parents=True, exist_ok=True)
    onnx_path = folder / "ensemble.onnx"
    # export with 2 windows so the window count can change on the phone
    torch.onnx.export(ens, (x[:2],), onnx_path, input_names=["audio"], output_names=["prob"],
                      dynamic_axes={"audio": {0: "windows"}, "prob": {0: "windows"}})
    single_file(onnx_path)
    # check onnx matches pytorch on 3 windows (not the 2 it was exported with)
    diff = float(np.abs(ort.InferenceSession(str(onnx_path)).run(None, {"audio": x.numpy()})[0]
                        - ens(x).detach().numpy()).max())
    assert diff < 1e-4, diff

    split = json.load(open(ROOT / "ml/out/split.json"))
    labels = load_labels()
    cv_recs = [r for r in labels if any(c.get("use_test") for c in cfgs) or split[r["pid"]] != "test"]
    runs = [json.load(open(f / "scores.json"))["oof"] for f in folders]
    oof = {st: {c: float(np.mean([r[st][c] for r in runs])) for c in ("clean", "phone_v2")}
           for st in set.intersection(*[set(r) for r in runs])}
    y, s = child_scores([r for r in cv_recs if r["stem"] in oof], oof, "phone_v2")
    # cutoff so `target` of the murmur kids are at or above it
    cutoff = float(np.sort(s[y])[int((1 - target) * y.sum())])
    report = dict(name=name, settings=[{k: v for k, v in c.items() if v != E.DEFAULTS[k]} for c in cfgs],
                  cutoff=cutoff,
                  cutoff_rule=f"catches {target:.0%} of murmur children in cross-validation (phone-like audio, top + bottom)",
                  cross_validation=dict(phone=at_cutoff(y, s, cutoff),
                                        clean=at_cutoff(*child_scores([r for r in cv_recs if r["stem"] in oof],
                                                                      oof, "clean"), cutoff)))

    # test kids only get scored once at the end (use --no-test while comparing candidates)
    ens_dev = ens.to(dev)
    if "--no-test" not in sys.argv and not any(c.get("use_test") for c in cfgs):
        test = [r for r in labels if split[r["pid"]] == "test"]
        xs = [load_circor(r["stem"]) for r in test]
        tscores = {}
        for cond in ("clean", "phone_v2"):
            p = E.score(_ProbAsLogit(ens_dev), xs, win, E.CONDITIONS[cond], dev, [r["spot"] for r in test])
            for r, v in zip(test, p):
                tscores.setdefault(r["stem"], {})[cond] = float(v)
        report["test"] = {c: {"children": {"app_spots_PV_TV": at_cutoff(*child_scores(test, tscores, k), cutoff)}}
                          for c, k in (("phone", "phone_v2"), ("clean", "clean"))}
    elif install and not any(c.get("use_test") for c in cfgs):
        sys.exit("--install needs the test evaluation; drop --no-test")

    zen = E.zenodo_set("small")
    if zen:
        zs = {c: E.score(_ProbAsLogit(ens_dev), zen["x"], win, E.CONDITIONS[c], dev, zen["spots"])
              for c in ("clean", "phone_v2")}
        report["outside_cohorts"] = E.zenodo_metrics(zen, zs)
        for cohort in sorted({m["project_name"] for m in zen["items"]}):
            kids = {}
            for m, v in zip(zen["items"], zs["phone_v2"]):
                if m["project_name"] == cohort:
                    k = kids.setdefault(m["patient_id"], [False, -np.inf])
                    k[0], k[1] = k[0] or bool(m["has_murmur"]), max(k[1], float(v))
            yk, sk = np.array([k[0] for k in kids.values()]), np.array([k[1] for k in kids.values()])
            report["outside_cohorts"][f"{cohort}_phone_at_cutoff"] = at_cutoff(yk, sk, cutoff)
    pascal_x, pascal_y, our_paths, our_x = E.external_sets()
    report["pascal_auroc"] = round(auroc(pascal_y, E.score(_ProbAsLogit(ens_dev), pascal_x, win, None, dev)), 4)
    ours = dict(zip([p.stem for p in our_paths],
                    E.score(_ProbAsLogit(ens_dev), our_x, win, None, dev).round(3).tolist()))
    report["our_recordings"] = ours
    if zen:
        report["outside_cohort_scores"] = {c: [round(float(v), 4) for v in zs[c]] for c in zs}
    json.dump(report, open(folder / ("candidate_report.json" if "--no-test" in sys.argv else "final_report.json"), "w"),
              indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("our_recordings", "outside_cohort_scores")}, indent=2))

    if install:
        app = ROOT / "app/model"
        shutil.copy(onnx_path, app / "murmur.onnx")
        json.dump(dict(threshold=cutoff, sample_rate=SR, window=win, hop=win // 2, skip_seconds=2),
                  open(app / "config.json", "w"), indent=2)
        json.dump(dict(model=f"average of 5 cross-validated models ({name})", cutoff_rule=report["cutoff_rule"],
                       test=report.get("test"), outside_cohorts=report.get("outside_cohorts")),
                  open(app / "metrics.json", "w"), indent=2)
        print("installed into app/model")


class _ProbAsLogit(torch.nn.Module):
    """E.score applies a sigmoid but the ensemble already gives probabilities, so undo it."""

    def __init__(self, ens):
        super().__init__()
        self.ens = ens

    def forward(self, audio):
        p = self.ens(audio).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))


if __name__ == "__main__":
    main()
