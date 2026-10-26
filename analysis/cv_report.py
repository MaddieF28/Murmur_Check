"""Compares spot options and cutoffs on the cross-validated scores, then checks the test kids.
Run: .venv/bin/python analysis/cv_report.py [SPOTS, default PV+TV] [TARGET, default 0.85]
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from data import load_labels  # noqa: E402
from train import auroc  # noqa: E402

SPOTS = ("AV", "PV", "TV", "MV")
NAMES = {"AV": "aortic", "PV": "top", "TV": "bottom", "MV": "under pec"}
rng = np.random.default_rng(0)


def child_scores(kids, oof, spots, cond):
    """Highest score of each kid's recordings at `spots` (cond = "clean" or "phone")."""
    return np.array([max(oof[r["stem"]][cond] for r in k["recs"] if r["spot"] in spots) for k in kids])


def boot(y, *scores, n=2000):
    # skip resamples that are all one class
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].all() or not y[i].any():
            continue
        stats.append([auroc(y[i], s[i]) for s in scores])
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return [auroc(y, s) for s in scores], lo, hi, stats


def main():
    chosen = tuple(sys.argv[1].split("+")) if len(sys.argv) > 1 else ("PV", "TV")
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    oof = json.load(open(ROOT / "ml/out/cv/oof.json"))["oof"]
    labels = load_labels()
    kids = group_children(labels, oof)
    # only kids with all four spots so every option is compared fairly
    all4 = [k for k in kids.values() if {r["spot"] for r in k["recs"]} == set(SPOTS)]
    y = np.array([k["murmur"] for k in all4])
    print(f"{len(all4)} children recorded at all four spots, {y.sum()} with murmurs "
          f"(every child scored by a model that never trained on them)\n")

    options = [(s,) for s in SPOTS] + list(combinations(SPOTS, 2)) + [SPOTS]
    base = child_scores(all4, oof, chosen, "phone")
    print(f"{'spots':22} {'stethoscope AUROC':>20} {'phone-like AUROC':>20} "
          f"{'vs ' + '+'.join(chosen) + ' (phone)':>22}")
    for spots in options:
        clean = child_scores(all4, oof, spots, "clean")
        phone = child_scores(all4, oof, spots, "phone")
        (a_c,), lo_c, hi_c, _ = boot(y, clean)
        (a_p, a_b), lo, hi, st = boot(y, phone, base)
        d = st[:, 0] - st[:, 1]
        name = "+".join(NAMES[s] for s in spots) if len(spots) < 4 else "all four"
        print(f"{name:22} {a_c:.3f} ({lo_c[0]:.2f}-{hi_c[0]:.2f}) {a_p:9.3f} ({lo[0]:.2f}-{hi[0]:.2f}) "
              f"{a_p - a_b:+9.3f} ({np.percentile(d, 2.5):+.2f} to {np.percentile(d, 97.5):+.2f})")

    have = [k for k in kids.values() if set(chosen) <= {r["spot"] for r in k["recs"]}]
    yk = np.array([k["murmur"] for k in have])
    cutoff = None
    for cond in ("phone", "clean"):
        p = child_scores(have, oof, chosen, cond)
        print(f"\ncutoffs for {'+'.join(chosen)} ({len(have)} children, {yk.sum()} with murmurs), {cond}-like audio:")
        for t_sens in (0.70, 0.75, 0.80, 0.85, 0.90):
            t = float(np.sort(p[yk])[int((1 - t_sens) * yk.sum())])
            print(f"  catch {t_sens:.0%} of murmurs: cutoff {t:.3f}, "
                  f"clears {(p[~yk] < t).mean():.0%} of children without one")
            if cond == "phone" and t_sens == target:
                cutoff = t

    # one look at the untouched test kids
    test_path = ROOT / "ml/out/cv/test.json"
    if cutoff is None or not test_path.exists():
        return
    scores = json.load(open(test_path))
    tkids = [k for k in group_children(labels, scores).values()
             if set(chosen) <= {r["spot"] for r in k["recs"]}]
    yt = np.array([k["murmur"] for k in tkids])
    print(f"\ntest children ({len(tkids)}, {yt.sum()} with murmurs), ensemble, cutoff {cutoff:.3f}:")
    for cond in ("phone", "clean"):
        p = child_scores(tkids, scores, chosen, cond)
        (a,), lo, hi, _ = boot(yt, p)
        print(f"  {cond}-like: AUROC {a:.3f} ({lo[0]:.2f}-{hi[0]:.2f}), "
              f"caught {(p[yt] >= cutoff).sum()} of {yt.sum()} murmurs "
              f"({(p[yt] >= cutoff).mean():.0%}), cleared {(p[~yt] < cutoff).mean():.0%} of children without one")


def group_children(labels, scores):
    kids = {}
    for r in labels:
        if r["spot"] in SPOTS and r["stem"] in scores:
            kids.setdefault(r["pid"], {"murmur": r["child_murmur"], "recs": []})["recs"].append(r)
    return kids


if __name__ == "__main__":
    main()
