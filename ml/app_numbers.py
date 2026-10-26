"""Results for the installed model using the app's rule, on CV children and the held-out test children.
--install writes app/model/metrics.json.
Usage: .venv/bin/python ml/app_numbers.py NAME [--install]
"""
import json
import sys

import numpy as np
import torch

import experiments as E
from crossval import Ensemble, assign_folds
from data import load_circor, load_labels
from final_numbers import OPTION2, cutoff_rule, per_child
from finalize import _ProbAsLogit
from model import SR


def tally(s, y, cut):
    """caught, missed, cleared, false alarms"""
    f = s >= cut
    return np.array([(f & y).sum(), (~f & y).sum(), (~f & ~y).sum(), (f & ~y).sum()])


def plain(name, t):
    caught, missed, cleared, false_alarms = (int(v) for v in t)
    n = caught + missed + cleared + false_alarms
    return dict(name=name, children=n, murmur_children=caught + missed,
                sensitivity=round(caught / (caught + missed), 4),
                specificity=round(cleared / (cleared + false_alarms), 4),
                accuracy=round((caught + cleared) / n, 4))


def main():
    name = sys.argv[1]
    folder = E.EXP / name if (E.EXP / name).exists() else E.ROOT / "ml/out/exp_pc" / name
    cfg = dict(E.DEFAULTS, **json.load(open(folder / "config.json")))
    oof = json.load(open(folder / "scores.json"))["oof"]
    split = json.load(open(E.ROOT / "ml/out/split.json"))
    labels = load_labels()
    cv = [r for r in labels if split[r["pid"]] != "test"]
    fold = assign_folds(cv)
    entries = [(r["pid"], r["child_murmur"], r["spot"], oof[r["stem"]]["phone_v2"])
               for r in cv if r["stem"] in oof]

    # each fold's line comes from the other 4 folds so no kid sets their own line
    t_cv = np.zeros(4, int)
    for k in range(5):
        s, y = per_child([e for e in entries if fold[e[0]] == k], OPTION2)
        t_cv += tally(s, y, cutoff_rule([e for e in entries if fold[e[0]] != k]))
    cutoff = cutoff_rule(entries)

    # test kids: average the 5 fold nets (= the app's model) and score once
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    nets = []
    for k in range(5):
        m = E.build_model(cfg)
        m.load_state_dict(torch.load(folder / f"fold{k}.pt", map_location="cpu"))
        nets.append(m.eval())
    ens = _ProbAsLogit(Ensemble(nets).eval().to(dev))
    test = [r for r in labels if split[r["pid"]] == "test"]
    p = E.score(ens, [load_circor(r["stem"]) for r in test], int(cfg["win_s"] * SR),
                E.CONDITIONS["phone_v2"], dev, [r["spot"] for r in test])
    s, y = per_child([(r["pid"], r["child_murmur"], r["spot"], float(v)) for r, v in zip(test, p)], OPTION2)
    t_test = tally(s, y, cutoff)

    groups = [plain("Cross-validation children", t_cv), plain("Test children, never used for anything", t_test)]
    every = plain("All", t_cv + t_test)
    for g in groups + [every]:
        print(f"{g['name']:40} {g['children']:4d} ({g['murmur_children']:3d}): murmur -> echo {g['sensitivity']:.0%}, "
              f"no murmur -> no echo {g['specificity']:.0%}, right {g['accuracy']:.0%}")
    print(f"the app's line: {cutoff:.4f}")
    if "--install" in sys.argv:
        json.dump(dict(model=name, cutoff=cutoff,
                       tested_on=f"{every['children']} children from the CirCor collection in Brazil",
                       every_child=every, groups=groups),
                  open(E.ROOT / "app/model/metrics.json", "w"), indent=2)
        print("wrote app/model/metrics.json")


if __name__ == "__main__":
    main()
