"""Checks how close each phone simulation is to our real iPhone recordings.
AUROC near 0.5 means a classifier can't tell them apart (good).
Run: .venv/bin/python analysis/check_phone_match.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from data import (augment, load_circor, load_labels, load_phone, phone_v2,  # noqa: E402
                  simulate_phone, windows)
from model import SR  # noqa: E402

# 12 log-spaced bands, 25-1900 Hz
EDGES = np.geomspace(25, 1900, 13)
FRAME = 400  # 100 ms frames


def features(w):
    """Spectrum shape + loud-vs-quiet range per band, for each window."""
    spec = np.abs(np.fft.rfft(w.reshape(len(w), -1, FRAME) * np.hanning(FRAME), axis=2)) ** 2
    f = np.fft.rfftfreq(FRAME, 1 / SR)
    bands = np.stack([spec[:, :, (f >= a) & (f < b)].sum(2) for a, b in zip(EDGES, EDGES[1:])],
                     axis=2) + 1e-20
    # subtract overall level so loudness doesn't matter
    shape = 10 * np.log10(bands.mean(1))
    shape -= shape.mean(1, keepdims=True)
    lb = 10 * np.log10(bands)
    spread = np.percentile(lb, 90, axis=1) - np.percentile(lb, 10, axis=1)
    return np.concatenate([shape, spread], axis=1)


def main():
    ours = [p for spot in ("PV", "TV", "MV")
            for p in sorted((ROOT / "recordings").glob(f"*_me_*_{spot}.wav"))]
    real = [(windows(load_phone(p)), p.stem.split("_")[-1]) for p in ours]
    split = json.load(open(ROOT / "ml/out/split.json"))
    random.seed(0)
    pool = [r for r in load_labels() if split[r["pid"]] != "test" and r["spot"] in ("PV", "TV", "MV")]
    circor = [(windows(load_circor(r["stem"])), r["spot"]) for r in random.sample(pool, 120)]
    gen = torch.Generator().manual_seed(0)
    sims = {
        "no simulation (stethoscope as is)": lambda w, s: w,
        "v1 fixed (current evaluation)": lambda w, s: simulate_phone(w, gen),
        "v1 random (current training)": lambda w, s: augment(w, gen),
        "v2 fixed (matched to our phone)": lambda w, s: phone_v2(w, gen, [s] * len(w), jitter=False),
        "v2 random (matched, with jitter)": lambda w, s: phone_v2(w, gen, [s] * len(w), jitter=True),
    }
    xr = np.concatenate([features(w) for w, _ in real])
    gr = np.concatenate([[i] * len(w) for i, (w, _) in enumerate(real)])
    print(f"{len(ours)} of our recordings ({len(xr)} windows) vs 120 CirCor recordings")
    for name, sim in sims.items():
        xs = [features(sim(torch.from_numpy(w), s).numpy()) for w, s in circor]
        gs = np.concatenate([[1000 + i] * len(x) for i, x in enumerate(xs)])
        x = np.concatenate([xr, np.concatenate(xs)])
        y = np.r_[np.ones(len(xr)), np.zeros(len(gs))]
        groups = np.r_[gr, gs]
        # group by recording so windows from one file don't leak into test
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(x, y, groups):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
            pred[te] = clf.fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
        print(f"  {name:36s} real-vs-simulated AUROC {roc_auc_score(y, pred):.3f}")
        sim_x = np.concatenate(xs)
        nb = len(EDGES) - 1
        diff = xr.mean(0) - sim_x.mean(0)
        print("      real minus simulated, shape dB: " + " ".join(f"{v:+.0f}" for v in diff[:nb]))
        print("      real minus simulated, range dB: " + " ".join(f"{v:+.0f}" for v in diff[nb:]))
    print("  bands (Hz): " + " ".join(f"{np.sqrt(a * b):.0f}" for a, b in zip(EDGES, EDGES[1:])))


if __name__ == "__main__":
    main()
