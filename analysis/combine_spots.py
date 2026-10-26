"""Tries different spot sets and ways of combining spot scores, same 85% cutoff rule for each.
Run: .venv/bin/python analysis/combine_spots.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
import experiments as E  # noqa: E402
from data import load_labels  # noqa: E402
from train import auroc  # noqa: E402

FINAL = "pc5_v2_gru_noise"  # model the app uses

# PV = top, TV = bottom, MV = under pec, AV = aortic
SETS = {"top + bottom (app now)": ("PV", "TV"), "top + bottom + under pec": ("PV", "TV", "MV"),
        "all four": ("PV", "TV", "MV", "AV"), "top only": ("PV",), "bottom only": ("TV",),
        "under pec only": ("MV",)}

RULES = {
    "any spot over the line": lambda s: s.max(),
    "average": lambda s: s.mean(),
    "two highest averaged": lambda s: np.sort(s)[-2:].mean(),
    "combined chance": lambda s: 1 - np.prod(1 - s),
}
rng = np.random.default_rng(0)


def per_child(entries, spots):
    """Kids recorded at every spot in `spots`, with their scores by spot."""
    kids = {}
    for pid, y, spot, s in entries:
        if spot in spots:
            k = kids.setdefault(pid, dict(y=y, by_spot={}))
            k["by_spot"].setdefault(spot, []).append(s)
    kids = [k for k in kids.values() if set(k["by_spot"]) >= set(spots)]
    return kids


def scores(kids, rule):
    # highest per spot first, then combine
    return np.array([RULES[rule](np.array([max(v) for v in k["by_spot"].values()])) for k in kids]), \
        np.array([k["y"] for k in kids])


def main():
    split = json.load(open(ROOT / "ml/out/split.json"))
    oof = json.load(open(ROOT / f"ml/out/exp_pc/{FINAL}/scores.json"))["oof"]
    circor = [(r["pid"], r["child_murmur"], r["spot"], oof[r["stem"]]["phone_v2"])
              for r in load_labels()
              if split[r["pid"]] != "test" and r["stem"] in oof]
    report = json.load(open(ROOT / f"ml/out/exp/final_{FINAL}/final_report.json"))
    zen = E.zenodo_set("small")
    names = {"ULSB": "PV", "LLSB": "TV", "APEX": "MV", "URSB": "AV"}
    outside = {c: [(m["patient_id"], bool(m["has_murmur"]),
                    names.get(str(m["recording_location"]).upper()), s)
                   for m, s in zip(zen["items"], report["outside_cohort_scores"]["phone_v2"])
                   if m["project_name"] == c]
               for c in ("WC2023", "WVU")}

    print("Held-out children, phone-like audio. Each row's cutoff catches 85% of murmurs in CirCor cross-validation.")
    print(f"{'spots':26} {'how they are combined':24} {'CirCor':>16} {'WC2023':>22} {'WVU':>22}")
    print(f"{'':26} {'':24} {'AUROC  clears':>16} {'AUROC catches clears':>22} {'AUROC catches clears':>22}")
    base = {}
    for set_name, spots in SETS.items():
        for rule in (RULES if len(spots) > 1 else ["any spot over the line"]):
            kids = per_child(circor, spots)
            s, y = scores(kids, rule)
            cut = float(np.sort(s[y])[int(0.15 * y.sum())])
            row = f"{set_name:26} {rule:24} {auroc(y, s):6.3f} {(s[~y] < cut).mean():7.0%}  "
            for c in ("WC2023", "WVU"):
                ks = per_child(outside[c], spots)
                so, yo = scores(ks, rule)
                row += f"{auroc(yo, so):7.3f} {(so[yo] >= cut).mean():7.0%} {(so[~yo] < cut).mean():6.0%}  "
            print(row + f"(n={len(kids)} CirCor children)")
            base[(set_name, rule)] = (kids, rule)

    # does the third spot actually help? paired bootstrap
    kids3 = per_child(circor, ("PV", "TV", "MV"))
    y = np.array([k["y"] for k in kids3])
    two = np.array([max(max(k["by_spot"]["PV"]), max(k["by_spot"]["TV"])) for k in kids3])
    three = np.array([max(max(v) for v in k["by_spot"].values()) for k in kids3])
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(auroc(y[i], three[i]) - auroc(y[i], two[i]))
    print(f"\nAdding 'under pec' to top + bottom ('any spot' rule), same {len(y)} CirCor children: "
          f"AUROC {auroc(y, two):.3f} -> {auroc(y, three):.3f}, "
          f"difference 95% range {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}")


if __name__ == "__main__":
    main()
