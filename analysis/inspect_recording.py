"""Plots a phone recording next to a CirCor one to see if the beats come through.
Run: python3 analysis/inspect_recording.py recordings/<file>.wav [circor_record_id]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parent.parent
FS = 2000  # heart sounds are well under 1 kHz


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= np.abs(x).max() or 1
    return resample_poly(x, FS, sr) if sr != FS else x


def bandpass(x, lo=25, hi=400):
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), x)


def envelope(x):
    hop = FS // 100
    e = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    return np.convolve(e, np.ones(7) / 7, mode="same")


def periodicity(env):
    """How regularly the loudness repeats (0-1), and the heart rate it matches."""
    # skip first 2 s (placing the phone), clip so one bump can't dominate
    e = env[200:]
    e = np.minimum(e, np.percentile(e, 95))
    z = e - e.mean()
    r = np.correlate(z, z, mode="full")[len(z) - 1:]
    r /= r[0] or 1
    lags = np.arange(33, 151)  # 40-180 bpm
    k = lags[np.argmax(r[lags])]
    return r[k], 6000 / k  # 60 s * 100 values per second


def describe(name, x):
    f, p = welch(x, fs=FS, nperseg=4096)
    band = lambda a, b: p[(f >= a) & (f < b)].sum() / p.sum()
    per, bpm = periodicity(envelope(bandpass(x)))
    print(f"{name}: periodicity {per:.2f}, ~{bpm:.0f} bpm | power <25 Hz {band(0, 25):.0%}, "
          f"25-150 {band(25, 150):.0%}, 150-400 {band(150, 400):.0%}, >400 {band(400, 1000):.0%}")
    return f, p


def main():
    phone = Path(sys.argv[1])
    ref_id = sys.argv[2] if len(sys.argv) > 2 else None
    refs = sorted((ROOT / "data/circor/training_data").glob(f"{ref_id or '*'}_MV.wav"))
    series = [(phone.stem, load(phone))]
    if refs:
        series.append((f"CirCor {refs[0].stem}", load(refs[0])))

    fig, axes = plt.subplots(len(series), 3, figsize=(16, 3.2 * len(series)), squeeze=False)
    for row, (name, x) in zip(axes, series):
        f, p = describe(name, x)
        t = np.arange(len(x)) / FS
        sel = (t > 1) & (t < 7)
        row[0].plot(t[sel], x[sel], lw=.4)
        row[0].set_title(f"{name}: raw")
        row[1].plot(t[sel], bandpass(x)[sel], lw=.4, color="C3")
        row[1].set_title("25-400 Hz")
        row[2].semilogy(f, p)
        row[2].set_xlim(0, 800)
        row[2].set_title("spectrum")
    fig.tight_layout()
    out = phone.with_suffix(".png")
    fig.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    main()
