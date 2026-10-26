"""Train the murmur model on CirCor, test on held-out children, export ONNX to ml/out/.
Usage: .venv/bin/python ml/train.py [--epochs 40]"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.stats import rankdata
from torch import nn

from data import augment, load_circor, load_labels, simulate_phone, split_patients, to_float, windows
from model import SR, WIN, ExportedModel, MurmurNet

OUT = Path(__file__).resolve().parent / "out"
APP_SPOTS = {"PV", "TV"}  # top and bottom spots the app uses


def auroc(y, p):
    y, r = np.asarray(y, bool), rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sens_spec(y, p, t):
    y, flag = np.asarray(y, bool), np.asarray(p) >= t
    return float(flag[y].mean()), float((~flag[~y]).mean())


def by_child(recs, probs, spots=None):
    """One score per child = their highest recording (same rule as the app)."""
    kids = {}
    for r, p in zip(recs, probs):
        if spots is None or r["spot"] in spots:
            has, best = kids.get(r["pid"], (r["child_murmur"], 0.0))
            kids[r["pid"]] = (has, max(best, p))
    y, p = zip(*kids.values())
    return np.array(y), np.array(p)


@torch.no_grad()
def predict(model, audio, phone=False, dev="cpu"):
    model.eval()
    # fixed seed so the phone noise is the same every time
    gen = torch.Generator().manual_seed(0)
    out = []
    for x in audio:
        w = torch.from_numpy(windows(x))
        if phone:
            w = simulate_phone(w, gen)
        out.append(torch.sigmoid(model(w.to(dev))).mean().item())
    return np.array(out)


def fit(train_recs, audio, dev, args, val=None, save_to=None):
    """Train a new model; with val, keeps the epoch with the best val AUROC."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)
    y = np.array([r["label"] for r in train_recs])
    pos_weight = float((y == 0).sum() / (y == 1).sum())  # murmurs are rarer so weight them up
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    model = MurmurNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    steps = math.ceil(len(train_recs) * args.crops / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps)

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        # new random 4 s crops from every recording each epoch
        items = [
            (r, rng.integers(0, len(audio[r["stem"]]) - WIN + 1))
            for r in train_recs for _ in range(args.crops)]
        rng.shuffle(items)
        for b in range(0, len(items), args.batch):
            batch = items[b:b + args.batch]
            x = torch.from_numpy(np.stack([to_float(audio[r["stem"]][o:o + WIN]) for r, o in batch]))
            target = torch.tensor([float(r["label"]) for r, _ in batch], device=dev)
            loss = loss_fn(model(augment(x, gen).to(dev)), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        entry, line = dict(epoch=epoch, loss=float(np.mean(losses))), f"epoch {epoch:2d}  loss {np.mean(losses):.3f}"
        if val:
            y_val, a = [r["label"] for r in val], [audio[r["stem"]] for r in val]
            clean, phone = (
                auroc(y_val, predict(model, a, dev=dev)),
                auroc(y_val, predict(model, a, phone=True, dev=dev)))
            entry.update(val_auroc=clean, val_auroc_phone=phone)
            line += f"  val AUROC {clean:.3f}  phone-like {phone:.3f}"
            if (clean + phone) / 2 > best:
                best, line = (clean + phone) / 2, line + " *"
                torch.save(model.state_dict(), save_to)
        history.append(entry)
        print(f"{line}  {time.time() - t0:.0f}s", flush=True)

    if val:
        model.load_state_dict(torch.load(save_to))
    return model, history


def evaluate(model, recs, split, audio, dev, target_sens):
    """Pick the cutoff on val children, then report results on test children."""
    probs = {}
    for part in ("val", "test"):
        rs = [r for r in recs if split[r["pid"]] == part]
        a = [audio[r["stem"]] for r in rs]
        probs[part] = (rs, {c: predict(model, a, phone=c == "phone", dev=dev) for c in ("clean", "phone")})

    # cutoff that catches target_sens of val murmur kids (app spots, phone-like audio)
    rs, p = probs["val"]
    yk, pk = by_child(rs, p["phone"], APP_SPOTS)
    threshold = float(np.sort(pk[yk])[int((1 - target_sens) * yk.sum())])

    rs, p = probs["test"]
    report = {}
    for cond in ("clean", "phone"):
        lab = [i for i, r in enumerate(rs) if r["label"] is not None]
        rec = {"auroc": auroc([rs[i]["label"] for i in lab], p[cond][lab]), "auroc_by_spot": {}}
        for spot in ("AV", "PV", "TV", "MV"):
            idx = [i for i in lab if rs[i]["spot"] == spot]
            rec["auroc_by_spot"][spot] = auroc([rs[i]["label"] for i in idx], p[cond][idx])
        kids = {}
        for name, spots in (("app_spots_PV_TV", APP_SPOTS), ("all_spots", None)):
            yk, pk = by_child(rs, p[cond], spots)
            sens, spec = sens_spec(yk, pk, threshold)
            kids[name] = dict(children=len(yk), murmur_children=int(yk.sum()), auroc=auroc(yk, pk),
                              sensitivity=sens, specificity=spec)
        report[cond] = dict(recordings=rec, children=kids)
    return threshold, report


def single_file(path):
    """Merge the exporter's separate weights file back in so the app loads one file."""
    onnx.save(onnx.load(path), path)
    Path(f"{path}.data").unlink(missing_ok=True)


def export(model):
    exported = ExportedModel(model.cpu().eval()).eval()
    x = torch.from_numpy(windows(np.random.default_rng(0).standard_normal(3 * WIN).astype(np.float32) * 0.1))
    path = OUT / "murmur.onnx"
    # dynamic_axes so the app can send any number of windows
    torch.onnx.export(exported, (x[:2],), path, input_names=["audio"], output_names=["prob"],
                      dynamic_axes={"audio": {0: "windows"}, "prob": {0: "windows"}})
    single_file(path)
    got = ort.InferenceSession(str(path)).run(None, {"audio": x.numpy()})[0]
    diff = float(np.abs(got - exported(x).detach().numpy()).max())
    print(f"ONNX check on {len(x)} windows: max difference from PyTorch {diff:.2e}")
    assert diff < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--target-sens", type=float, default=0.85)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    recs = load_labels()
    split = split_patients(recs)
    audio = {r["stem"]: load_circor(r["stem"]) for r in recs}
    print(f"{len(recs)} recordings from {len(split)} children, training on {dev}", flush=True)

    train_recs = [r for r in recs if split[r["pid"]] == "train" and r["label"] is not None]
    val = [r for r in recs if split[r["pid"]] == "val" and r["label"] is not None]
    model, history = fit(train_recs, audio, dev, args, val=val, save_to=OUT / "model.pt")
    threshold, report = evaluate(model, recs, split, audio, dev, args.target_sens)
    export(model)

    json.dump(split, open(OUT / "split.json", "w"), indent=0)
    json.dump(dict(threshold=threshold, args=vars(args), test=report, history=history),
              open(OUT / "metrics.json", "w"), indent=2)
    # the app needs these to cut audio into windows the same way
    json.dump(dict(threshold=threshold, sample_rate=SR, window=WIN, hop=WIN // 2, skip_seconds=2),
              open(OUT / "config.json", "w"), indent=2)
    print(f"\nthreshold {threshold:.3f} (catches {args.target_sens:.0%} of validation murmur children)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
