"""Average spectrum by source and spot, in dB vs CirCor.
Run: .venv/bin/python analysis/compare_spectra.py
"""
import random
from pathlib import Path

import numpy as np
from phone_vs_stethoscope import ROOT, mean_spectrum

HZ = [30, 50, 75, 100, 150, 200, 300, 400, 600]  # frequencies shown in the table


def main():
    random.seed(0)
    circor = ROOT / "data/circor/training_data"
    rec = sorted((ROOT / "recordings").glob("*.wav"))
    # (files, seconds to skip at the start)
    groups = {f"CirCor {s}": (random.sample(sorted(circor.glob(f"*_{s}.wav")), 150), 0)
              for s in ("PV", "TV", "MV")}
    for s, name in (("PV", "top"), ("TV", "bottom"), ("MV", "under pec")):
        groups[f"our phone, {name}"] = ([p for p in rec if p.stem.endswith(s)], 2)
    for k in ("normal", "murmur"):
        groups[f"PASCAL iPhone {k}"] = (sorted((ROOT / f"data/pascal/{k}").glob("*.wav")), 1)

    f, ref = mean_spectrum(random.sample(sorted(circor.glob("*.wav")), 300))
    print(f"{'source':24} {'n':>3} " + " ".join(f"{h:>6}" for h in HZ) + "   (dB vs CirCor average)")
    for name, (paths, skip) in groups.items():
        _, p = mean_spectrum(paths, skip_s=skip)
        cells = " ".join(f"{10 * np.log10(p[np.argmin(abs(f - h))] / ref[np.argmin(abs(f - h))]):+6.1f}"
                         for h in HZ)
        print(f"{name:24} {len(paths):3d} {cells}")


if __name__ == "__main__":
    main()
