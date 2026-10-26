"""Tests every spot and pair of spots on the val/test kids with the exported model.
Run: .venv/bin/python analysis/spot_options.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from data import load_circor, load_labels, simulate_phone, windows  # noqa: E402
from train import auroc  # noqa: E402

OUT = ROOT / "ml/out"
SPOTS = ("AV", "PV", "TV", "MV")


def score_recordings(recs):
    """Mean model score per recording, clean and phone-like."""
    session = ort.InferenceSession(str(OUT / "murmur.onnx"))
    gen = torch.Generator().manual_seed(0)
    probs = {}
    for r in recs:
        w = windows(load_circor(r["stem"]))
        phone = simulate_phone(torch.from_numpy(w), gen).numpy().astype(np.float32)
        probs[r["stem"]] = {c: float(session.run(None, {"audio": x})[0].mean())
                            for c, x in (("clean", w), ("phone", phone))}
    return probs


def main():
    split = json.load(open(OUT / "split.json"))
    threshold = json.load(open(OUT / "config.json"))["threshold"]
    kids = {}
    for r in load_labels():
        if split[r["pid"]] in ("val", "test") and r["spot"] in SPOTS:
            kids.setdefault(r["pid"], {"murmur": r["child_murmur"], "part": split[r["pid"]],
                                       "recs": []})["recs"].append(r)
    # only kids with all four spots
    kids = {p: k for p, k in kids.items() if {r["spot"] for r in k["recs"]} == set(SPOTS)}

    # scoring is slow so cache it
    cache = OUT / "probs_val_test.json"
    if cache.exists():
        probs = json.load(open(cache))
    else:
        probs = score_recordings([r for k in kids.values() for r in k["recs"]])
        json.dump(probs, open(cache, "w"))

    for part in ("test", "val"):
        group = [k for k in kids.values() if k["part"] == part]
        y = np.array([k["murmur"] for k in group])
        print(f"\n{part}: {len(group)} children recorded at all four spots, {y.sum()} with murmurs; "
              f"cutoff {threshold:.2f}")
        print(f"{'spots':12} {'stethoscope AUROC':>18} {'phone-like AUROC':>17} {'caught':>8} {'cleared':>8}")
        options = [(s,) for s in SPOTS] + list(combinations(SPOTS, 2)) + [SPOTS]
        for spots in options:
            row = []
            for cond in ("clean", "phone"):
                p = np.array([max(probs[r["stem"]][cond] for r in k["recs"] if r["spot"] in spots)
                              for k in group])
                row.append((auroc(y, p), (p[y] >= threshold).mean(), (p[~y] < threshold).mean()))
            print(f"{'+'.join(spots):12} {row[0][0]:18.3f} {row[1][0]:17.3f} {row[1][1]:8.0%} {row[1][2]:8.0%}")
        print("(caught/cleared: phone-like audio at the current cutoff)")


if __name__ == "__main__":
    main()
