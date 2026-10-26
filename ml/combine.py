"""Average finished experiments together. metadata=1 also checks if age + sex help.
Usage: .venv/bin/python ml/combine.py EXP1 EXP2 ... [metadata=1]
"""
import csv
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data import CIRCOR, load_labels
from experiments import APP_SPOTS, EXP, ROOT, child_auroc
from train import auroc

AGES = ["Neonate", "Infant", "Child", "Adolescent", "Young Adult"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main():
    names = [a for a in sys.argv[1:] if "=" not in a]
    use_meta = "metadata=1" in sys.argv[1:]
    # exp_pc = runs copied over from the PC
    runs = [json.load(open(EXP / n / "scores.json" if (EXP / n).exists() else ROOT / "ml/out/exp_pc" / n / "scores.json"))
            for n in names]
    stems = set.intersection(*[set(r["oof"]) for r in runs])
    conds = set.intersection(*[set(next(iter(r["oof"].values()))) for r in runs])
    oof = {s: {c: float(np.mean([logit(r["oof"][s][c]) for r in runs])) for c in conds} for s in stems}

    split = json.load(open(ROOT / "ml/out/split.json"))
    recs = [r for r in load_labels() if split[r["pid"]] != "test" and r["stem"] in stems]
    print(f"{' + '.join(names)}: {len({r['pid'] for r in recs})} children")
    for c in sorted(conds):
        print(f"  {c:9s} app spots {child_auroc(recs, oof, c, APP_SPOTS):.4f}   all spots {child_auroc(recs, oof, c):.4f}")
    pascal_y = np.array([p["label"] for p in json.load(open(ROOT / "ml/out/pascal_accepted.json"))])
    pascal = np.mean([logit(np.array(r["pascal"])) for r in runs], axis=0)
    print(f"  PASCAL real iPhone AUROC {auroc(pascal_y, pascal):.4f}")
    ours = {s: float(np.mean([logit(r["ours"][s]) for r in runs if s in r["ours"]])) for s in runs[0]["ours"]}
    for spot in ("PV", "TV", "MV", "THIGH", "AIR"):
        v = [1 / (1 + np.exp(-x)) for s, x in ours.items() if "_me_" in s and s.endswith(f"_{spot}")]
        if v:
            print(f"  ours {spot:5s} mean {np.mean(v):.3f}  (n={len(v)})")

    if use_meta:
        meta = {r["Patient ID"]: r for r in csv.DictReader(open(CIRCOR / "training_data.csv"))}
        for c in ("phone_v2", "clean"):
            if c not in conds:
                continue
            # one score per kid = their max at the app spots
            kids = {}
            for r in recs:
                if r["spot"] in APP_SPOTS:
                    k = kids.setdefault(r["pid"], dict(y=r["child_murmur"], spots=set(), s=-1e9))
                    k["spots"].add(r["spot"])
                    k["s"] = max(k["s"], oof[r["stem"]][c])
            pids = [p for p, k in kids.items() if k["spots"] >= APP_SPOTS]
            y = np.array([kids[p]["y"] for p in pids])
            audio_only = np.array([[kids[p]["s"]] for p in pids])
            m = [meta[p] for p in pids]
            extra = np.array([[a == x["Age"] for a in AGES] + [x["Sex"] == "Female"] for x in m], dtype=float)
            for label, x in (("audio only", audio_only), ("audio + age + sex", np.hstack([audio_only, extra]))):
                pred = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
                    pred[te] = LogisticRegression(C=1.0, max_iter=1000).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
                print(f"  {c:9s} {label:18s} child AUROC {auroc(y, pred):.4f}")


if __name__ == "__main__":
    main()
