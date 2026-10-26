"""Table of every experiment (Mac + PC) ranked by one column.
Usage: .venv/bin/python ml/results.py [sort column, default cv_phone_v2]
"""
import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
COLUMNS = [("cv_clean", "steth"), ("cv_phone_v2", "v2"), ("cv_phone_v2_allspots", "v2 all"),
           ("zen_WVU_phone_v2", "WVU v2"), ("zen_WC2023_phone_v2", "WC v2"), ("zen_WVU_clean", "WVU"),
           ("zen_WC2023_clean", "WC"), ("pascal_beyond_quality", "P>qual"), ("me_PV", "me top"),
           ("me_TV", "me bot"), ("me_THIGH", "thigh"), ("minutes", "min")]


def load():
    """Every finished experiment, with rescore.py numbers filled in."""
    rows = []
    if (OUT / "exp/results.csv").exists():
        rows += [dict(r, settings=json.loads(r["settings"]), where="mac")
                 for r in csv.DictReader(open(OUT / "exp/results.csv"))]
    for where, path in (("mac", OUT / "exp/results.jsonl"), ("pc", OUT / "exp_pc/results.jsonl")):
        if path.exists():
            rows += [dict(json.loads(line), where=where) for line in open(path) if line.strip()]
    latest = {(r["where"], r["name"]): r for r in rows}  # newest copy wins
    rescored = {}
    if (OUT / "exp/rescored.jsonl").exists():
        for line in open(OUT / "exp/rescored.jsonl"):
            if line.strip():
                r = json.loads(line)
                rescored.setdefault(r["name"], {}).update(r)
    out = []
    for r in latest.values():
        if r.get("failed"):
            continue
        # rescored numbers only fill in gaps
        for k, v in rescored.get(r["name"], {}).items():
            if r.get(k) is None:
                r[k] = v
        out.append(r)
    return out


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "cv_phone_v2"
    num = lambda v: float(v) if v not in (None, "") else float("nan")
    rows = sorted(load(), key=lambda r: -num(r.get(key)) if r.get(key) not in (None, "") else 0)
    print(f"{'name':22} {'where':5} " + " ".join(f"{h:>6}" for _, h in COLUMNS) + "  settings")
    for r in rows:
        cells = " ".join((f"{num(r.get(c)):6.0f}" if c == "minutes" else f"{num(r.get(c)):6.3f}")
                         if r.get(c) not in (None, "") else f"{'-':>6}" for c, _ in COLUMNS)
        settings = {k: v for k, v in r["settings"].items() if k != "folds"}
        print(f"{r['name']:22} {r['where']:5} {cells}  {json.dumps(settings)}")


if __name__ == "__main__":
    main()
