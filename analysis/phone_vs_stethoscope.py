"""Phone vs CirCor stethoscope: average spectra matched at 40-80 Hz, difference in dB.
Run: .venv/bin/python analysis/phone_vs_stethoscope.py recordings/*.wav
"""
import random
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parent.parent
SR = 4000  # CirCor's rate


def mean_spectrum(paths, skip_s=0.0):
    """Average spectrum of `paths`, each scaled by its own 40-80 Hz level. Also used by compare_spectra.py."""
    out = []
    for path in paths:
        sr, x = wavfile.read(path)
        x = x[int(skip_s * sr):].astype(np.float64)
        if sr != SR:
            x = resample_poly(x, SR, sr)
        if len(x) < 2 * SR:
            continue
        f, p = welch(x, fs=SR, nperseg=1024)
        out.append(p / p[(f >= 40) & (f < 80)].mean())
    return f, np.mean(out, axis=0)


def main():
    phone = sys.argv[1:]
    random.seed(0)
    circor = random.sample(sorted((ROOT / "data/circor/training_data").glob("*.wav")), 300)
    f, steth = mean_spectrum(circor)
    _, ph = mean_spectrum(phone, skip_s=2)  # skip placing the phone
    print(f"phone ({len(phone)} recordings) vs stethoscope, dB, matched at 40-80 Hz:")
    for hz in [30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 600]:
        i = np.argmin(abs(f - hz))
        print(f"  {hz:4d} Hz  {10 * np.log10(ph[i] / steth[i]):+6.1f}")


if __name__ == "__main__":
    main()
