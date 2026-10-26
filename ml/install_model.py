"""Copies the CV ensemble into the app with its cutoff and test results.
Usage: .venv/bin/python ml/install_model.py [--target 0.85]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from cv_report import child_scores, group_children  # noqa: E402
from data import load_labels  # noqa: E402
from model import SR, WIN  # noqa: E402
from train import auroc  # noqa: E402

CV = ROOT / "ml/out/cv"
APP = ROOT / "app/model"
SPOTS = ("PV", "TV")


def with_spots(kids):
    return [k for k in kids.values() if set(SPOTS) <= {r["spot"] for r in k["recs"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.85)
    args = ap.parse_args()
    labels = load_labels()

    # cutoff only comes from the CV scores, never the test kids
    oof = json.load(open(CV / "oof.json"))["oof"]
    kids = with_spots(group_children(labels, oof))
    y = np.array([k["murmur"] for k in kids])
    p = child_scores(kids, oof, SPOTS, "phone")
    cutoff = float(np.sort(p[y])[int((1 - args.target) * y.sum())])

    test = json.load(open(CV / "test.json"))
    tkids = with_spots(group_children(labels, test))
    yt = np.array([k["murmur"] for k in tkids])
    results = {}
    for cond in ("phone", "clean"):
        pt = child_scores(tkids, test, SPOTS, cond)
        results[cond] = {"children": {"app_spots_PV_TV": dict(
            children=len(tkids), murmur_children=int(yt.sum()), auroc=auroc(yt, pt),
            sensitivity=float((pt[yt] >= cutoff).mean()), specificity=float((pt[~yt] < cutoff).mean()))}}

    shutil.copy(CV / "ensemble.onnx", APP / "murmur.onnx")
    json.dump(dict(threshold=cutoff, sample_rate=SR, window=WIN, hop=WIN // 2, skip_seconds=2),
              open(APP / "config.json", "w"), indent=2)
    json.dump(dict(model="average of 5 cross-validated models",
                   cutoff_rule=f"catches {args.target:.0%} of murmur children in cross-validation "
                               "(phone-like audio, top + bottom spots)",
                   test=results), open(APP / "metrics.json", "w"), indent=2)
    print(f"installed ensemble, cutoff {cutoff:.3f}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
