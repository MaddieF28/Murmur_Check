"""Results table for leave-one-dataset-out (lodo_) or k-fold (kfold_) rounds, per dataset + averaged.
Usage: .venv/bin/python ml/lodo_report.py [--prefix lodo_] [--target 0.85] [--model final_all] [--write-app] [--also kfold_]
"""
import json
import sys
from pathlib import Path

import numpy as np

import datasets
from train import auroc

OUT = Path(__file__).resolve().parent / "out"
ROOT = OUT.parent.parent
NAMES = {"circor": "CirCor (Brazil)", "wvu": "WVU (US heart clinic)", "wc2023": "WC2023 (US heart clinic)",
         "ohh": "OHH-2025 (US hospital)", "zch": "ZCHSound (China)", "pascalb": "PASCAL B (Brazil)",
         "pascala": "PASCAL A (real iPhones)"}


def run_folder(name):
    return next((p for p in (OUT / "exp_pc" / name, OUT / "exp" / name) if p.exists()), OUT / "exp_pc" / name)


def line_for(s, y, target):
    """cutoff that sends `target` of the murmur kids for an echo"""
    return float(np.sort(s[y])[int((1 - target) * y.sum())])


def held_out(prefix, recs, cond):
    """dataset -> (scores, labels, round) from the round that left each kid out, plus unfinished rounds"""
    prefixes = prefix.split("+")
    keys = [str(k) for k in range(5)] if prefixes[0].startswith("kfold") else list(NAMES)
    held, missing = {}, []
    for key in keys:
        runs = []
        for p in prefixes:
            path = run_folder(f"{p}{key}") / "scores.json"
            if path.exists() and path.stat().st_size > 2:  # "{}" = round not done yet
                runs.append(json.load(open(path)))
            else:
                missing.append(f"{p}{key}")
        if len(runs) == len(prefixes):
            held[key] = {st: {c: float(np.mean([r[st][c] for r in runs])) for c in runs[0][st]} for st in runs[0]}
    kids = {}
    for d in NAMES:
        parts = []
        for name, scores in held.items():
            rs = [r for r in recs if r["dataset"] == d and r["stem"] in scores]
            if rs:
                s, y = datasets.child_scores(rs, scores, cond)
                parts.append((s, y, np.full(len(y), name)))
        if parts:
            kids[d] = tuple(np.concatenate(p) for p in zip(*parts))
    return kids, missing


def evaluate(kids, target):
    s_all, y_all, g_all = (np.concatenate([k[i] for k in kids.values()]) for i in range(3))
    # each group's line only comes from the other groups
    lines = {g: line_for(s_all[g_all != g], y_all[g_all != g], target) if (g_all != g).any() else 0.5
             for g in set(g_all)}
    measure = lambda s, y, f: dict(sens=f[y].mean(), spec=(~f[~y]).mean(), acc=(f == y).mean(), auc=auroc(y, s))
    rows = {d: measure(s, y, s >= np.array([lines[x] for x in g])) for d, (s, y, g) in kids.items()}
    pooled = measure(s_all, y_all, s_all >= np.array([lines[x] for x in g_all]))
    return rows, pooled, line_for(s_all, y_all, target)


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    prefix, model, app = (arg("--prefix", "lodo_"), arg("--model", "final_all"),
                          Path(arg("--app-dir", ROOT / "app/model")))
    target, kfold = float(arg("--target", 0.85)), prefix.startswith("kfold")
    recs, _ = datasets.load(list(NAMES), with_audio=False)
    kids, missing = held_out(prefix, recs, "phone_v2")
    if not kids:
        sys.exit(f"no finished rounds for {prefix}")
    rows, pooled, app_cut = evaluate(kids, target)

    if kfold:
        print("Cross-validation: each child tested by a model that never heard them, trained on the rest of")
        print("every dataset (phone-like audio). Each fold's line is chosen from the other folds only.")
    else:
        print("Each dataset tested by a model trained on every OTHER dataset (phone-like audio).")
        print("Each dataset's line is chosen from the other datasets only.")
    print(f"Line aims to send {target:.0%} of children with murmurs for an echo.\n")
    print(f"{'dataset':28} {'children':>11} {'murmur -> echo':>15} {'no murmur -> no echo':>21} "
          f"{'total right':>12} {'ranking':>8}")
    for d, r in rows.items():
        y = kids[d][1]
        print(f"{NAMES[d]:28} {len(y):5d} ({y.sum():3d}) {r['sens']:14.0%} {r['spec']:20.0%} "
              f"{r['acc']:11.0%} {r['auc']:8.3f}")
    avg = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}
    print(f"{'AVERAGE over ' + str(len(rows)) + ' datasets':28} {'':>11} {avg['sens']:14.0%} {avg['spec']:20.0%} "
          f"{avg['acc']:11.0%} {avg['auc']:8.3f}")
    n = sum(len(k[1]) for k in kids.values())
    print(f"{'EVERY CHILD COUNTED ONCE':28} {n:5d} ({sum(k[1].sum() for k in kids.values()):3d}) {pooled['sens']:14.0%} "
          f"{pooled['spec']:20.0%} {pooled['acc']:11.0%} {pooled['auc']:8.3f}")
    print(f"\nThe app's line from every held-out score: {app_cut:.3f}")
    if missing:
        print("still running:", ", ".join(missing))
    if "--write-app" in sys.argv and not missing:
        c = json.load(open(app / "config.json"))
        c["threshold"] = app_cut
        json.dump(c, open(app / "config.json", "w"), indent=2)
        plain = lambda r: dict(sensitivity=round(float(r["sens"]), 4), specificity=round(float(r["spec"]), 4),
                               accuracy=round(float(r["acc"]), 4), ranking=round(float(r["auc"]), 4))
        per = {d: dict(name=NAMES[d], children=int(len(kids[d][1])), murmur_children=int(kids[d][1].sum()), **plain(r))
               for d, r in rows.items()}
        out = dict(model=model, cutoff=app_cut, target=target, rounds=prefix, test="children" if kfold else "datasets",
                   average=plain(avg), every_child=dict(children=n, **plain(pooled)), datasets=per)
        if "--also" in sys.argv:
            also = arg("--also", "")
            k2, missing2 = held_out(also, recs, "phone_v2")
            assert not missing2, missing2
            rows2, pooled2, _ = evaluate(k2, target)
            avg2 = {k: float(np.mean([r[k] for r in rows2.values()])) for k in next(iter(rows2.values()))}
            out["new_hospitals"] = dict(rounds=also, average=plain(avg2), every_child=plain(pooled2))
        json.dump(out, open(app / "metrics.json", "w"), indent=2)
        print("wrote the app's line and results")


if __name__ == "__main__":
    main()
