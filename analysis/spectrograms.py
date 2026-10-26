"""Stacked 4 s spectrograms (25-400 Hz) to eyeball murmur energy between S1 and S2.
Run: .venv/bin/python analysis/spectrograms.py out.png FILE[:phone] ...
  (FILE = phone .wav or a CirCor stem like 50782_TV; :phone applies the phone simulation)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from data import load_circor, load_phone, simulate_phone, to_float  # noqa: E402
from model import SR  # noqa: E402


def load(spec):
    name, _, mode = spec.partition(":")
    if name.endswith(".wav"):
        return Path(name).stem, load_phone(name)
    x = to_float(load_circor(name))
    if mode == "phone":
        x = simulate_phone(torch.from_numpy(x[None]), torch.Generator().manual_seed(0))[0].numpy()
    return f"CirCor {name}{' (phone-like)' if mode else ''}", x


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 2.4 * len(specs)), squeeze=False)
    for ax, spec in zip(axes[:, 0], specs):
        title, x = load(spec)
        x = x[SR:5 * SR]  # 4 s
        f, t, p = spectrogram(x, fs=SR, nperseg=256, noverlap=216)
        band = (f >= 25) & (f <= 400)
        logp = 10 * np.log10(p[band] + 1e-12)
        # subtract each frequency's mean so the color shows changes over time
        ax.pcolormesh(t, f[band], logp - logp.mean(axis=1, keepdims=True),
                      shading="auto", cmap="magma", vmin=-15, vmax=20)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Hz")
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print("wrote", out)


if __name__ == "__main__":
    main()
