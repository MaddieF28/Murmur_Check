"""Picks the murmur + normal kids for the app's "Try a sample" buttons and copies them to app/samples.
Usage: .venv/bin/python ml/pick_samples.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from data import load_labels, windows

ROOT = Path(__file__).resolve().parent.parent
LOC = {"ULSB": "PV", "LLSB": "TV", "APEX": "MV"}
SOURCES = {"CirCor": "CirCor DigiScope Phonocardiogram Dataset, PhysioNet (ODC-By 1.0)",
           "WVU": "Saini & Araujo, Zenodo 23460250 (CC BY 4.0)", "WC2023": "Saini & Araujo, Zenodo 23460250 (CC BY 4.0)"}


def heartbeat(x, fs=4000, skip_s=2):
    """same as beatClarity in app/app.js -> (clarity, bpm)"""
    sos = np.vstack([butter(2, 25, "highpass", fs=fs, output="sos")] * 2
                    + [butter(2, 400, "lowpass", fs=fs, output="sos")] * 2)
    f, hop = sosfilt(sos, x.astype(np.float64)), fs // 100
    env = np.sqrt((f[:len(f) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    env = np.convolve(env, np.ones(7), "same") / np.convolve(np.ones_like(env), np.ones(7), "same")
    e = env[skip_s * 100:]
    e = np.minimum(e, np.sort(e)[int(len(e) * 0.95)])
    z = e - e.mean()
    # lags of 0.32-1.52 s, about 40-180 bpm
    ac = {lag: (z[:-lag] * z[lag:]).sum() / ((z * z).sum() or 1) for lag in range(32, 152)}
    peaks = [lag for lag in range(33, 151) if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1]]
    return max(ac[lag] for lag in range(33, 151)), 6000 / max(peaks, key=ac.get) if peaks else 0


def candidates():
    kids, zen = {}, ROOT / "data/zenodo"
    for m in json.load(open(zen / "labels.json")).values():
        spot = LOC.get(str(m["recording_location"]).upper())
        if spot:
            k = kids.setdefault((m["project_name"], m["patient_id"]), dict(murmur=bool(m["has_murmur"]), recs={}))
            k["recs"][spot] = zen / m["project_name"] / f"{m['uuid']}.wav"
    for r in load_labels():
        if r["spot"] in LOC.values():
            k = kids.setdefault(("CirCor", r["pid"]), dict(murmur=r["child_murmur"], recs={}))
            k["recs"].setdefault(r["spot"], ROOT / "data/circor/training_data" / f"{r['stem']}.wav")
    return {key: k for key, k in kids.items() if set(k["recs"]) == set(LOC.values())}


def main():
    cutoff = json.load(open(ROOT / "app/model/config.json"))["threshold"]
    session = ort.InferenceSession(str(ROOT / "app/model/murmur.onnx"))
    scored = []
    for (cohort, pid), k in candidates().items():
        beats = [heartbeat(wavfile.read(p)[1]) for p in k["recs"].values()]
        rates = [bpm for _, bpm in beats]
        # want clear beats and about the same bpm at every spot
        if min(c for c, _ in beats) < 0.35 or min(rates) == 0 or max(rates) / min(rates) > 1.15:
            continue
        probs = {s: float(session.run(None, {"audio": windows(wavfile.read(p)[1][2 * 4000:])})[0].mean())
                 for s, p in k["recs"].items()}
        scored.append((cohort, pid, k, probs))
    murmur = max((c for c in scored if c[2]["murmur"]), key=lambda c: max(c[3].values()))
    normal = min((c for c in scored if not c[2]["murmur"]), key=lambda c: max(c[3].values()))
    dest = ROOT / "app/samples"
    for old in dest.glob("*.wav"):
        old.unlink()
    for kind, (cohort, pid, k, probs) in (("murmur", murmur), ("normal", normal)):
        for spot, path in k["recs"].items():
            shutil.copy(path, dest / f"{kind}_{spot}.wav")
        print(f"{kind}: {cohort} child {pid}, scores " + ", ".join(f"{s} {p:.2f}" for s, p in probs.items())
              + f" (cutoff {cutoff:.2f})")
    json.dump({kind: dict(child=f"{c[0]} child {c[1]}", source=SOURCES[c[0]])
               for kind, c in (("murmur", murmur), ("normal", normal))},
              open(dest / "samples.json", "w"), indent=2)


if __name__ == "__main__":
    main()
