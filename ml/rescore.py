"""Re-scores saved fold models on the outside cohorts + my recordings without retraining.
Usage: .venv/bin/python ml/rescore.py NAME [NAME ...]
"""
import json
import sys

import numpy as np
import torch

from experiments import CONDITIONS, DEFAULTS, EXP, ROOT, build_model, external_sets, score, zenodo_metrics, zenodo_set
from model import SR


def rescore(name, zen, our_paths, our_x, dev):
    folder = EXP / name if (EXP / name).exists() else ROOT / "ml/out/exp_pc" / name
    cfg = dict(DEFAULTS, **json.load(open(folder / "config.json")))
    win = int(cfg["win_s"] * SR)
    models = []
    for pt in sorted(folder.glob("fold*.pt")):
        m = build_model(cfg)
        m.load_state_dict(torch.load(pt, map_location="cpu"))
        models.append(m.to(dev))
    row = dict(name=name)
    if zen:
        zs = {c: np.mean([score(m, zen["x"], win, CONDITIONS[c], dev, zen["spots"]) for m in models], axis=0)
              for c in ("clean", "phone_v2")}
        row.update(zenodo_metrics(zen, zs))
    if our_x:
        ours = dict(zip([p.stem for p in our_paths], np.mean([score(m, our_x, win, None, dev) for m in models], axis=0)))
        # THIGH and AIR are just noise checks
        for spot in ("PV", "TV", "MV", "THIGH", "AIR"):
            v = [s for st, s in ours.items() if "_me_" in st and st.endswith(f"_{spot}")]
            row[f"me_{spot}"] = round(float(np.mean(v)), 3) if v else None
    with open(EXP / "rescored.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    zen = zenodo_set("small")
    _, _, our_paths, our_x = external_sets()
    for name in sys.argv[1:]:
        rescore(name, zen, our_paths, our_x, dev)


if __name__ == "__main__":
    main()
