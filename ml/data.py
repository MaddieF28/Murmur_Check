"""CirCor labels, patient split, audio windows, and phone-style augmentation."""
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from model import SR, WIN

ROOT = Path(__file__).resolve().parent.parent
CIRCOR = ROOT / "data/circor"


def load_labels():
    """One dict per recording. label is None if the child has a murmur but not at this spot."""
    patients = {r["Patient ID"]: r for r in csv.DictReader(open(CIRCOR / "training_data.csv"))}
    recs = []
    for line in open(CIRCOR / "RECORDS"):
        stem = Path(line.strip()).name
        pid, spot = stem.split("_")[:2]
        murmur = patients[pid]["Murmur"]
        if murmur == "Unknown":
            continue
        heard = patients[pid]["Murmur locations"].split("+")
        label = 0 if murmur == "Absent" else 1 if spot in heard else None
        recs.append(dict(
            stem=stem, pid=pid, spot=spot, label=label,
            child_murmur=murmur == "Present"))
    return recs


def split_patients(recs, seed=0):
    """70/15/15 train/val/test split by child, not by recording."""
    rng = random.Random(seed)
    split = {}
    for has_murmur in (True, False):
        pids = sorted({r["pid"] for r in recs if r["child_murmur"] == has_murmur})
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            split[pid] = (
                "train" if i < 0.7 * len(pids)
                else "val" if i < 0.85 * len(pids)
                else "test")
    return split


def load_circor(stem):
    sr, x = wavfile.read(CIRCOR / "training_data" / f"{stem}.wav")
    assert sr == SR, (stem, sr)
    return x


def load_phone(path, skip_s=2.0):
    """Load a phone recording at 4 kHz, skipping the first couple seconds."""
    sr, x = wavfile.read(path)
    x = x[int(skip_s * sr):].astype(np.float32)
    # gcd gives the smallest up/down ratio, e.g. 48000 -> 4000 is 1 up, 12 down
    g = math.gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def to_float(x):
    return x.astype(np.float32) / 32768 if x.dtype == np.int16 else x.astype(np.float32)


def windows(x, win=WIN, hop=None):
    x, hop = to_float(x), hop or win // 2
    if len(x) < win:
        x = np.resize(x, win)
    return np.stack([x[s:s + win] for s in range(0, len(x) - win + 1, hop)])


def phone_filter(x, cutoff, order, rumble, snr_db, gen=None):
    """Make stethoscope audio sound like a phone: low-pass, add rumble, add hiss."""
    batch, n = x.shape
    f = torch.fft.rfftfreq(n, 1 / SR)
    # butterworth low-pass done in the frequency domain
    x = torch.fft.irfft(torch.fft.rfft(x) * (1 + (f / cutoff) ** (2 * order)) ** -0.5, n=n)
    level = x.pow(2).mean(1, keepdim=True).sqrt() + 1e-8
    # rumble = noise low-passed at 15 Hz, like the phone moving on skin
    low = torch.fft.irfft(
        torch.fft.rfft(torch.randn(batch, n, generator=gen)) * (1 + (f / 15) ** 4) ** -0.5,
        n=n)
    x = x + low / (low.pow(2).mean(1, keepdim=True).sqrt() + 1e-8) * level * rumble
    return x + torch.randn(batch, n, generator=gen) * level * 10 ** (-snr_db / 20)


def augment(x, gen=None):
    """Random phone-like filtering for training (40-400 Hz cutoffs cover what our iPhone measured)."""
    batch = x.shape[0]
    u = lambda lo, hi: torch.empty(batch, 1).uniform_(lo, hi, generator=gen)
    cutoff = torch.exp(u(math.log(40), math.log(400)))
    # 1e6 Hz cutoff = no filter, so 20% stay as stethoscope sound
    cutoff = torch.where(
        torch.rand(batch, 1, generator=gen) < 0.8,
        cutoff,
        torch.full_like(cutoff, 1e6))
    order = torch.randint(1, 4, (batch, 1), generator=gen).float()
    return phone_filter(x, cutoff, order, u(0, 3), u(10, 40), gen)


def simulate_phone(x, gen=None):
    """Fixed phone filter for evaluation (70 Hz, 2nd order)."""
    full = lambda v: torch.full((x.shape[0], 1), float(v))
    return phone_filter(x, full(70), full(2), full(1), full(25), gen)


def random_eq(x, gen=None, max_db=6.0):
    """Random smooth EQ per window, to fake different phones and chest contact."""
    batch, n = x.shape
    f = torch.fft.rfftfreq(n, 1 / SR)
    logf = torch.log(f.clamp(min=20))
    curves = []
    for _ in range(batch):
        k = int(torch.randint(3, 7, (1,), generator=gen))
        knots = torch.linspace(math.log(20), math.log(2000), k)
        gains = torch.empty(k).uniform_(-max_db, max_db, generator=gen)
        curves.append(
            torch.from_numpy(np.interp(logf.numpy(), knots.numpy(), gains.numpy())).float())
    return torch.fft.irfft(torch.fft.rfft(x) * 10 ** (torch.stack(curves) / 20), n=n)


_profile = None


def phone_v2(x, gen=None, spots=None, jitter=True):
    """Make stethoscope audio sound like our iPhone, using the measured profile from analysis/phone_profile.py."""
    global _profile
    if _profile is None:
        _profile = dict(np.load(ROOT / "ml/out/phone_profile.npz"))
    prof, (batch, n) = _profile, x.shape

    have = [s for s in ("PV", "TV", "MV") if f"transfer_db_{s}" in prof]
    pick = torch.randint(0, len(have), (batch,), generator=gen).tolist()
    chosen = [
        s if spots is not None and s in have else have[i]
        for s, i in zip(spots or [None] * batch, pick)]

    f = torch.fft.rfftfreq(n, 1 / SR)
    tdb = torch.tensor(np.stack([np.interp(f.numpy(), prof["freqs"], prof[f"transfer_db_{s}"]) for s in chosen]),
                       dtype=torch.float32)
    df = SR / 1024  # the profile's Welch bin width
    level = torch.tensor([float(prof[f"heart_power_{s}"]) * df for s in chosen])[:, None]
    if jitter:
        u = lambda lo, hi: torch.empty(batch, 1).uniform_(lo, hi, generator=gen)
        tdb = tdb + u(-6, 6) * (f / 1000).clamp(max=1)
        level = level * 10 ** (u(-6, 6) / 10)

    # apply phone gain, then match the heart band to the measured loudness
    spec = torch.fft.rfft(x) * 10 ** (tdb / 20)
    band = (f >= 25) & (f <= 400)
    var = 2 * (spec.abs() ** 2)[:, band].sum(1, keepdim=True) / n ** 2
    y = torch.fft.irfft(spec * torch.sqrt(level / (var + 1e-20)), n=n)

    # add a slice of real phone noise (recorded on the thigh, no heart)
    bank = torch.from_numpy(prof["noise"])
    if len(bank) < 2 * n:
        bank = bank.repeat(2 * n // len(bank) + 1)
    starts = torch.randint(0, len(bank) - n, (batch,), generator=gen)
    noise = torch.stack([bank[s:s + n] for s in starts.tolist()])
    if jitter:
        noise = noise * 10 ** (torch.empty(batch, 1).uniform_(-3, 3, generator=gen) / 20)
    return y + noise
