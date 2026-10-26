"""Builds the phone profile (per-spot transfer vs CirCor + a thigh noise bank) for the phone simulation.
Run: .venv/bin/python analysis/phone_profile.py   (writes ml/out/phone_profile.npz)
"""
import random
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from data import load_phone, to_float, load_circor  # noqa: E402
from model import SR  # noqa: E402

NPERSEG = 1024
SPOTS = ("PV", "TV", "MV")  # top, bottom, under pec


def ours(spot):
    return [p for p in sorted((ROOT / "recordings").glob(f"*_me_*_{spot}.wav"))]


def psd(x):
    f, p = welch(x, fs=SR, nperseg=NPERSEG)
    return f, p


def main():
    # thigh has no heartbeat, so it's just the noise of a phone on skin
    noise_paths = ours("THIGH")
    noise = [load_phone(p) / 32768 for p in noise_paths]
    f, p_noise = psd(np.concatenate(noise))
    band = (f >= 25) & (f <= 400)
    profile = dict(freqs=f, noise=np.concatenate(noise).astype(np.float32), noise_psd=p_noise)
    print(f"noise bank: {len(noise_paths)} thigh recordings, {sum(map(len, noise)) / SR:.0f} s")

    random.seed(0)
    for spot in SPOTS:
        paths = ours(spot)
        if not paths:
            continue
        p_chest = np.mean([psd(load_phone(p) / 32768)[1] for p in paths], axis=0)
        p_heart = np.maximum(p_chest - p_noise, 0.05 * p_chest)
        circor = random.sample(sorted((ROOT / "data/circor/training_data").glob(f"*_{spot}.wav")), 200)
        refs = []
        for c in circor:
            _, p = psd(to_float(load_circor(c.stem)))
            refs.append(p / p[band].sum())  # same loudness so the average is just shape
        # phone vs stethoscope at each frequency, zeroed at 40-80 Hz
        transfer_db = 10 * np.log10(p_heart / np.mean(refs, axis=0))
        transfer_db -= transfer_db[(f >= 40) & (f <= 80)].mean()
        snr = 10 * np.log10(p_heart[band].sum() / p_noise[band].sum())
        profile[f"transfer_db_{spot}"] = transfer_db
        profile[f"heart_power_{spot}"] = p_heart[band].sum()
        print(f"{spot}: {len(paths)} recordings, heart vs thigh noise in 25-400 Hz {snr:+.1f} dB; transfer dB at "
              + ", ".join(f"{h} Hz {transfer_db[np.argmin(abs(f - h))]:+.0f}"
                          for h in (100, 150, 200, 300, 400, 600)))
    np.savez(ROOT / "ml/out/phone_profile.npz", **profile)


if __name__ == "__main__":
    main()
