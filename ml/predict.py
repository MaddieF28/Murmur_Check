"""Runs the exported model on phone recordings like the app does.
Usage: .venv/bin/python ml/predict.py recordings/*.wav
"""
import json
import sys
from pathlib import Path

import onnxruntime as ort

from data import load_phone, windows

OUT = Path(__file__).resolve().parent / "out"


def main():
    cfg = json.load(open(OUT / "config.json"))
    session = ort.InferenceSession(str(OUT / "murmur.onnx"))
    print(f"threshold {cfg['threshold']:.2f}")
    for path in sys.argv[1:]:
        # average over the windows, same as the app
        w = windows(load_phone(path, cfg["skip_seconds"]))
        probs = session.run(None, {"audio": w})[0]
        p = float(probs.mean())
        verdict = "murmur: get an echo" if p >= cfg["threshold"] else "no murmur heard"
        print(f"{Path(path).name:36s} {p:.2f}  (windows {probs.min():.2f}-{probs.max():.2f})  {verdict}")


if __name__ == "__main__":
    main()
