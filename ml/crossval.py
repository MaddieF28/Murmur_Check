"""5-fold cross-validation on the non-test children, then a 5-model ensemble for the app.
Usage: .venv/bin/python ml/crossval.py [--epochs 10]   (writes to ml/out/cv/)"""
import argparse
import json
import random

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from data import load_circor, load_labels, windows
from model import WIN, MurmurNet
from train import OUT, fit, predict, single_file

CV = OUT / "cv"
K = 5  # number of folds


def assign_folds(recs, seed=0):
    """Children dealt into K folds, murmur and no-murmur children shuffled separately."""
    rng = random.Random(seed)
    fold = {}
    for has_murmur in (True, False):
        pids = sorted({r["pid"] for r in recs if r["child_murmur"] == has_murmur})
        rng.shuffle(pids)
        # deal like cards so each fold gets about the same number of murmur kids
        for i, pid in enumerate(pids):
            fold[pid] = i % K
    return fold


class Ensemble(nn.Module):
    """Average murmur probability of the fold models."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, audio):
        return torch.stack([torch.sigmoid(n(audio)) for n in self.nets]).mean(dim=0)


def export_ensemble(nets):
    """Save the ensemble as one ONNX file and check it matches PyTorch."""
    ens = Ensemble([n.cpu().eval() for n in nets]).eval()
    x = torch.from_numpy(windows(np.random.default_rng(0).standard_normal(3 * WIN).astype(np.float32) * 0.1))
    path = CV / "ensemble.onnx"
    # dynamic_axes so the app can send any number of windows
    torch.onnx.export(ens, (x[:2],), path, input_names=["audio"], output_names=["prob"],
                      dynamic_axes={"audio": {0: "windows"}, "prob": {0: "windows"}})
    single_file(path)
    diff = float(np.abs(
        ort.InferenceSession(str(path)).run(None, {"audio": x.numpy()})[0]
        - ens(x).detach().numpy()).max())
    print(f"ensemble ONNX check: max difference from PyTorch {diff:.2e}", flush=True)
    assert diff < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    CV.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    split = json.load(open(OUT / "split.json"))
    all_recs = load_labels()
    recs = [r for r in all_recs if split[r["pid"]] != "test"]
    test = [r for r in all_recs if split[r["pid"]] == "test"]
    fold = assign_folds(recs)
    audio = {r["stem"]: load_circor(r["stem"]) for r in all_recs}
    oof, nets = {}, []  # out-of-fold scores
    for k in range(K):
        print(f"\nfold {k + 1} of {K}", flush=True)
        train_recs = [r for r in recs if fold[r["pid"]] != k and r["label"] is not None]
        net, _ = fit(train_recs, audio, dev, args)
        torch.save(net.state_dict(), CV / f"fold{k}.pt")
        held = [r for r in recs if fold[r["pid"]] == k]
        for cond in ("clean", "phone"):
            probs = predict(net, [audio[r["stem"]] for r in held], phone=cond == "phone", dev=dev)
            for r, p in zip(held, probs):
                oof.setdefault(r["stem"], {})[cond] = float(p)
        # save every fold in case the run dies
        json.dump(dict(fold=fold, oof=oof, args=vars(args)), open(CV / "oof.json", "w"))
        nets.append(net)
    export_ensemble(nets)

    a = [audio[r["stem"]] for r in test]
    scores = {
        c: np.mean([predict(n.to(dev), a, phone=c == "phone", dev=dev) for n in nets], axis=0)
        for c in ("clean", "phone")}
    json.dump(
        {r["stem"]: {c: float(scores[c][i]) for c in scores} for i, r in enumerate(test)},
        open(CV / "test.json", "w"))
    print(f"scored {len(test)} test recordings with the ensemble", flush=True)


if __name__ == "__main__":
    main()
