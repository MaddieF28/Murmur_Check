"""Final numbers for the app's rule (3 spots, any one over the cutoff = get an echo): k-fold + one-time OHH-2025 test.
Usage: .venv/bin/python ml/final_numbers.py NAME [--install]
"""
import json
import sys

import numpy as np
import torch

import experiments as E
from crossval import Ensemble, assign_folds
from data import load_labels
from finalize import _ProbAsLogit
from model import SR
from train import auroc

ROOT = E.ROOT
OPTION2 = ("PV", "TV", "MV")
# zenodo spot names -> ours
LOC = {"ULSB": "PV", "LLSB": "TV", "APEX": "MV", "URSB": "AV"}


def per_child(entries, spots):
    """(pid, murmur, spot, score) entries -> each kid's max score, only kids with every spot"""
    kids = {}
    for pid, y, spot, s in entries:
        if spot in spots:
            k = kids.setdefault(pid, [bool(y), {}])
            k[0] = k[0] or bool(y)
            k[1][spot] = max(k[1].get(spot, -1.0), s)
    kids = [k for k in kids.values() if set(k[1]) >= set(spots)]
    return np.array([max(k[1].values()) for k in kids]), np.array([k[0] for k in kids])


def plain(s, y, cut):
    flag = s >= cut
    return dict(children=int(len(y)), murmur_children=int(y.sum()),
                sensitivity=round(float(flag[y].mean()), 4),
                specificity=round(float((~flag[~y]).mean()), 4),
                accuracy=round(float((flag == y).mean()), 4),
                caught=int(flag[y].sum()), cleared=int((~flag[~y]).sum()))


def cutoff_rule(entries):
    """Cutoff that catches 85% of murmur kids using top + bottom spots."""
    s, y = per_child(entries, ("PV", "TV"))
    return float(np.sort(s[y])[int(0.15 * y.sum())])


def main():
    name = sys.argv[1]
    folder = E.EXP / name if (E.EXP / name).exists() else ROOT / "ml/out/exp_pc" / name
    cfg = dict(E.DEFAULTS, **json.load(open(folder / "config.json")))
    oof = json.load(open(folder / "scores.json"))["oof"]
    split = json.load(open(ROOT / "ml/out/split.json"))
    recs = [r for r in load_labels() if cfg["use_test"] or split[r["pid"]] != "test"]
    fold = assign_folds(recs)
    entries = [(r["pid"], r["child_murmur"], r["spot"], oof[r["stem"]]["phone_v2"])
               for r in recs if r["stem"] in oof]

    # each fold's cutoff is picked without that fold
    totals = np.zeros(4, int)
    for k in range(5):
        cut = cutoff_rule([e for e in entries if fold[e[0]] != k])
        s, y = per_child([e for e in entries if fold[e[0]] == k], OPTION2)
        f = s >= cut
        totals += [(f & y).sum(), (~f & y).sum(), (~f & ~y).sum(), (f & ~y).sum()]
    caught, missed, cleared, false_alarms = totals
    kfold = dict(children=int(totals.sum()), murmur_children=int(caught + missed),
                 sensitivity=round(caught / (caught + missed), 4),
                 specificity=round(cleared / (cleared + false_alarms), 4),
                 accuracy=round((caught + cleared) / totals.sum(), 4),
                 caught=int(caught), cleared=int(cleared))
    cutoff = cutoff_rule(entries)

    # OHH-2025 only gets scored once, after everything is already picked, so it stays a fair test
    from scipy.io import wavfile
    labels = json.load(open(ROOT / "data/zenodo/labels_OHH-2025.json"))
    items = list(labels.values())
    xs = [wavfile.read(ROOT / "data/zenodo/OHH-2025" / f"{m['uuid']}.wav")[1] for m in items]
    # unknown spot -> random PV/TV for the phone sim (seeded)
    rng = np.random.default_rng(0)
    spots = [LOC.get(str(m["recording_location"]).upper()) or str(rng.choice(["PV", "TV"])) for m in items]
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    nets = []
    for k in range(5):
        m = E.build_model(cfg)
        m.load_state_dict(torch.load(folder / f"fold{k}.pt", map_location="cpu"))
        nets.append(m.eval())
    ens = _ProbAsLogit(Ensemble(nets).eval().to(dev))
    win = int(cfg["win_s"] * SR)
    outside = {}
    for cond in ("phone_v2", "clean"):
        sc = E.score(ens, xs, win, E.CONDITIONS[cond], dev, spots)
        ent = [(m["patient_id"], m["has_murmur"], LOC.get(str(m["recording_location"]).upper()), float(p))
               for m, p in zip(items, sc)]
        s3, y3 = per_child(ent, OPTION2)
        every = {}
        for pid, yy, _, p in ent:
            e = every.setdefault(pid, [False, -1.0])
            e[0], e[1] = e[0] or bool(yy), max(e[1], p)
        ya, sa = np.array([e[0] for e in every.values()]), np.array([e[1] for e in every.values()])
        outside[cond] = dict(option2=plain(s3, y3, cutoff), option2_ranking=round(auroc(y3, s3), 4),
                             all_spots_ranking=round(auroc(ya, sa), 4))

    result = dict(name=name, cutoff=cutoff, kfold_option2=kfold, ohh_2025=outside)
    json.dump(result, open(folder / "final_numbers.json", "w"), indent=2)
    print(json.dumps(result, indent=2))
    if "--install" in sys.argv:
        o = outside["phone_v2"]["option2"]
        json.dump(dict(model=name, cutoff=cutoff, kfold=kfold,
                       outside=dict(name="OHH-2025",
                                    **{k: o[k] for k in ("children", "murmur_children", "sensitivity",
                                                         "specificity", "accuracy")})),
                  open(ROOT / "app/model/metrics.json", "w"), indent=2)
        print("wrote app/model/metrics.json")


if __name__ == "__main__":
    main()
