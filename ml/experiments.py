"""Train/score model variants with the same CV setup and log a row to ml/out/exp/results.jsonl.
Usage: .venv/bin/python ml/experiments.py NAME [key=value ...]
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import data
from crossval import assign_folds
from data import load_circor, load_labels, load_phone, to_float, windows
from model import SR, MurmurNet
from train import auroc

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "ml/out/exp"
# settings you can change with key=value (the default's type is used to parse it)
DEFAULTS = dict(
    epochs=10, lr=2e-3, batch=64, crops=4, win_s=4.0, wd=1e-3, dropout=0.3, mixup=0.0,
    widths="16,32,64,64", head="meanmax", lo=25, hi=400, bands=31, frame=256, hop=40, cmn=1,
    aug="v1", labels="spot", speed=0.0, fms=0.0, zenodo="small", extra="", noise_neg=0.0, balance=0,
    seeds=1, use_test=0, lodo="", datasets="circor,wvu,wc2023,ohh,zch,pascalb,pascala",
    seed=0, folds="0,1,2,3,4")
AUGMENT = {
    "none": lambda x, gen, spots: x,
    "v1": lambda x, gen, spots: data.augment(x, gen),
    "v2": lambda x, gen, spots: data.phone_v2(x, gen, spots),
    "mix": lambda x, gen, spots: (data.phone_v2(x, gen, spots) if torch.rand(1, generator=gen).item() < 0.5
                                  else data.augment(x, gen)),
    "v2eq": lambda x, gen, spots: data.phone_v2(data.random_eq(x, gen), gen, spots),
}
# how held-out audio gets scored
CONDITIONS = {
    "clean": None,
    "phone_v1": lambda x, gen, spots: data.simulate_phone(x, gen),
    "phone_v2": lambda x, gen, spots: data.phone_v2(x, gen, spots, jitter=False),
}
APP_SPOTS = {"PV", "TV"}


def parse(args):
    cfg = dict(DEFAULTS)
    for a in args:
        key, value = a.split("=", 1)
        assert key in DEFAULTS, f"unknown setting {key}"
        cfg[key] = type(DEFAULTS[key])(value)
    return cfg


def build_model(cfg):
    spec = dict(
        lo=cfg["lo"], hi=cfg["hi"], bands=cfg["bands"],
        frame=cfg["frame"], hop=cfg["hop"], cmn=bool(cfg["cmn"]))
    widths = tuple(int(w) for w in cfg["widths"].split(","))
    return MurmurNet(widths=widths, head=cfg["head"], dropout=cfg["dropout"], spec=spec, fms=cfg["fms"])


def crop(x, start, win, length=None):
    """Cut a window; if length != win it gets stretched/squeezed to win (speed change)."""
    length = length or win
    seg = to_float(x[start:start + length])
    if len(seg) < length:
        seg = np.resize(seg, length)
    if length != win:
        seg = np.interp(np.linspace(0, length - 1, win), np.arange(length), seg).astype(np.float32)
    return seg


def train_fold(cfg, recs, audio, dev):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    win = int(cfg["win_s"] * SR)
    y = np.array([r["label"] for r in recs])
    pick = None
    if cfg["balance"]:  # equal murmur/normal per dataset so the dataset can't give away the label
        ds = np.array([r.get("dataset", "circor") for r in recs])
        pick = np.zeros(len(recs))
        for d in set(ds):
            share = (ds == d).sum() ** (1 if cfg["balance"] == 1 else 0.5)
            for c in (0, 1):
                m = (ds == d) & (y == c)
                pick[m] = share / 2 / max(1, m.sum())
        pick /= pick.sum()
    pos_weight = 1.0 if cfg["balance"] else float((y == 0).sum() / (y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = math.ceil(len(recs) * cfg["crops"] / cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps)
    augment = AUGMENT[cfg["aug"]]
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, losses = time.time(), []
        items = []
        drawn = (np.repeat(np.arange(len(recs)), cfg["crops"]) if pick is None
                 else rng.choice(len(recs), size=len(recs) * cfg["crops"], p=pick))
        for i in drawn:
            r = recs[i]
            length = int(win * rng.uniform(1 - cfg["speed"], 1 + cfg["speed"]))
            items.append((r, rng.integers(0, max(1, len(audio[r["stem"]]) - length + 1)), length))
        rng.shuffle(items)
        for b in range(0, len(items), cfg["batch"]):
            batch = items[b:b + cfg["batch"]]
            x = torch.from_numpy(np.stack([crop(audio[r["stem"]], o, win, n) for r, o, n in batch]))
            t = torch.tensor([float(r["label"]) for r, _, _ in batch])
            if cfg["noise_neg"] > 0:  # silent windows = just phone noise, labeled 0
                silent = torch.from_numpy(rng.random(len(batch)) < cfg["noise_neg"])
                x[silent], t[silent] = 0.0, 0.0
            # real phone recordings don't get the phone simulation
            phone = torch.tensor([bool(r.get("phone")) for r, _, _ in batch])
            simulated = augment(x, gen, [r["spot"] for r, _, _ in batch])
            x = torch.where(phone[:, None], x, simulated) if phone.any() else simulated
            if cfg["mixup"] > 0:
                lam = float(np.random.default_rng(int(rng.integers(1 << 30))).beta(cfg["mixup"], cfg["mixup"]))
                perm = torch.randperm(len(x), generator=gen)
                x, t = lam * x + (1 - lam) * x[perm], lam * t + (1 - lam) * t[perm]
            loss = loss_fn(model(x.to(dev)), t.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"  epoch {epoch + 1:2d}  loss {np.mean(losses):.3f}  {time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def score(model, xs, win, cond, dev, spots=None, chunk=256):
    """Average murmur probability per recording, batching windows chunk at a time."""
    if not len(xs):
        return np.array([])
    model.eval()
    gen = torch.Generator().manual_seed(0)
    ws, owner = [], []
    for i, x in enumerate(xs):
        w = torch.from_numpy(windows(x, win=win))
        if cond is not None:
            w = cond(w, gen, [spots[i] if spots else None] * len(w))
        ws.append(w.float())
        owner += [i] * len(w)
    allw, owner = torch.cat(ws), np.array(owner)
    probs = torch.cat([
        torch.sigmoid(model(allw[j:j + chunk].to(dev))).cpu()
        for j in range(0, len(allw), chunk)]).numpy()
    return np.array([probs[owner == i].mean() for i in range(len(xs))])


def child_auroc(recs, scores, cond, spots=None):
    kids = {}
    for r in recs:
        if spots is None or r["spot"] in spots:
            kids.setdefault(r["pid"], [r["child_murmur"], set(), -np.inf])
            k = kids[r["pid"]]
            k[1].add(r["spot"])
            k[2] = max(k[2], scores[r["stem"]][cond])
    kids = [k for k in kids.values() if spots is None or k[1] >= spots]
    return auroc([k[0] for k in kids], [k[2] for k in kids])


SPOT_NAMES = {"ULSB": "PV", "LLSB": "TV", "APEX": "MV", "URSB": "AV", "PV": "PV", "TV": "TV", "MV": "MV", "AV": "AV"}


def fetch_cohorts_from_mac():
    """For the PC: download the extracted outside cohorts from the Mac (75 MB)."""
    import io
    import re
    import urllib.request
    import zipfile
    try:
        url = (ROOT / "pc/base_url.txt").read_text().strip() + "/pc/zenodo_small.zip"
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=600).read()))
    except Exception as e:
        print(f"couldn't fetch the outside cohorts: {e}", flush=True)
        return
    out = ROOT / "data/zenodo"
    for name in z.namelist():
        if name in ("labels.json", "sent_to_mac.txt") or re.fullmatch(r"(WVU|WC2023)/[A-Za-z0-9_.-]+\.wav", name):
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_bytes(z.read(name))


def zenodo_set(which):
    """Outside Zenodo cohorts. which = "none", "small" (WVU + WC2023), or anything else for + OHH-2025."""
    from scipy.io import wavfile
    path = ROOT / "data/zenodo/labels.json"
    if which != "none" and not path.exists() and (ROOT / "pc/base_url.txt").exists():
        fetch_cohorts_from_mac()
    if which == "none" or not path.exists():
        return None
    cohorts = ("WVU", "WC2023") if which == "small" else ("WVU", "WC2023", "OHH-2025")
    items = [m for m in json.load(open(path)).values() if m["project_name"] in cohorts]
    rng = np.random.default_rng(0)
    spot_of = lambda m: next((SPOT_NAMES[str(v).upper()] for k, v in m.items()
                              if k.lower() in ("recording_location", "location", "site", "spot")
                              and str(v).upper() in SPOT_NAMES), None)
    # no known spot -> random app spot for the phone sim
    spots = [spot_of(m) or str(rng.choice(["PV", "TV"])) for m in items]
    xs = [wavfile.read(ROOT / "data/zenodo" / m["project_name"] / f"{m['uuid']}.wav")[1] for m in items]
    return dict(x=xs, items=items, spots=spots)


def zenodo_metrics(zen, scores):
    out = {}
    for cohort in sorted({m["project_name"] for m in zen["items"]}):
        for cond, s in scores.items():
            app, every, rec_y, rec_s = {}, {}, [], []
            for m, spot, p in zip(zen["items"], zen["spots"], s):
                if m["project_name"] != cohort:
                    continue
                y = bool(m["has_murmur"])
                k = every.setdefault(m["patient_id"], [y, -np.inf])
                k[1] = max(k[1], float(p))
                if str(m.get("recording_location", "")).upper() in ("ULSB", "LLSB"):
                    a = app.setdefault(m["patient_id"], [y, -np.inf, set()])
                    a[1] = max(a[1], float(p))
                    a[2].add(spot)
                if m.get("murmur_heard_this_location") is not None:
                    rec_y.append(bool(m["murmur_heard_this_location"]))
                    rec_s.append(float(p))
            both = [a for a in app.values() if a[2] >= APP_SPOTS]
            if both:
                out[f"zen_{cohort}_{cond}"] = round(auroc([a[0] for a in both], [a[1] for a in both]), 4)
            out[f"zen_{cohort}_{cond}_all"] = round(
                auroc([k[0] for k in every.values()], [k[1] for k in every.values()]), 4)
            if rec_y and 0 < sum(rec_y) < len(rec_y):
                out[f"zen_{cohort}_{cond}_rec"] = round(auroc(rec_y, rec_s), 4)
    return out


def external_sets():
    pascal = json.load(open(ROOT / "ml/out/pascal_accepted.json"))
    ours = [p for p in sorted((ROOT / "recordings").glob("*.wav")) if p.stem != "1024170520_iphone_bare_MV"]
    return ([load_phone(ROOT / p["path"]) for p in pascal], np.array([p["label"] for p in pascal]),
            ours, [load_phone(p) for p in ours])


def extra_training(names):
    """Extra labeled recordings used only for training (pascalb, zch)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    out = []
    if "pascalb" in names:
        for kind, label in (("murmur", 1), ("normal", 0)):
            for p in sorted((ROOT / "data/pascal_b" / kind).glob("*.wav")):
                x, sr = sf.read(p, dtype="int16")
                assert sr == SR, (p, sr)
                out.append(dict(stem=f"pascalb_{p.stem}", spot=None, label=label, audio=x))
    if "zch" in names:
        import csv
        base = ROOT / "data/zchsound/ZCHSound"
        labels = {"VSD": 1, "PDA": 1, "NORMAL": 0}
        for r in csv.DictReader(open(base / "Clean Heartsound Data Details.csv", encoding="utf-8-sig")):
            if r["diagnosis"].strip() in labels:
                x, sr = sf.read(base / "clean Heartsound Data" / r["fileName"].strip(), dtype="int16")
                x = np.clip(resample_poly(x.astype(np.float32), SR, sr), -32768, 32767).astype(np.int16)
                out.append(dict(
                    stem=f"zch_{r['fileName'].strip()}", spot="PV",
                    label=labels[r["diagnosis"].strip()], audio=x))
    return out


def run(name, cfg):
    """Run one experiment: train and score every fold, then log it."""
    out = EXP / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    t_start = time.time()
    if cfg["lodo"]:
        return run_lodo(name, cfg, out, dev, t_start)

    split = json.load(open(ROOT / "ml/out/split.json"))
    # use_test=1 is for the final model only (trains on test kids too)
    recs = [r for r in load_labels() if cfg["use_test"] or split[r["pid"]] != "test"]
    if cfg["labels"] == "child":
        recs = [dict(r, label=int(r["child_murmur"])) for r in recs]
    fold = assign_folds(recs)
    audio = {r["stem"]: load_circor(r["stem"]) for r in recs}
    extra = extra_training(cfg["extra"].split(",")) if cfg["extra"] else []
    audio.update({e["stem"]: e.pop("audio") for e in extra})
    win = int(cfg["win_s"] * SR)
    pascal_x, pascal_y, our_paths, our_x = external_sets()
    zen = zenodo_set(cfg["zenodo"])
    zen_scores = {"clean": [], "phone_v2": []}

    folds = [int(f) for f in cfg["folds"].split(",")]
    oof, pascal_scores, our_scores = {}, [], []
    for k in folds:
        print(f"{name}: fold {k}", flush=True)
        model = train_fold(
            cfg, [r for r in recs if fold[r["pid"]] != k and r["label"] is not None] + extra, audio, dev)
        torch.save(model.state_dict(), out / f"fold{k}.pt")
        held = [r for r in recs if fold[r["pid"]] == k]
        for cond, fn in CONDITIONS.items():
            probs = score(model, [audio[r["stem"]] for r in held], win, fn, dev, [r["spot"] for r in held])
            for r, p in zip(held, probs):
                oof.setdefault(r["stem"], {})[cond] = float(p)
        pascal_scores.append(score(model, pascal_x, win, None, dev))
        our_scores.append(score(model, our_x, win, None, dev))
        if zen:
            for cond in zen_scores:
                zen_scores[cond].append(score(model, zen["x"], win, CONDITIONS[cond], dev, zen["spots"]))

    pascal = np.mean(pascal_scores, axis=0)
    ours = dict(zip([p.stem for p in our_paths], np.mean(our_scores, axis=0).round(4).tolist()))
    extra = zenodo_metrics(zen, {c: np.mean(s, axis=0) for c, s in zen_scores.items()}) if zen else {}
    log_result(name, out, cfg["folds"], time.time() - t_start, recs, oof, pascal, pascal_y, ours,
               {k: v for k, v in cfg.items() if v != DEFAULTS[k]}, extra)


def run_lodo(name, cfg, out, dev, t_start, cutoff=0.5):
    """Leave one dataset out (or lodo=fold0..4 for child folds, lodo=none for the final model)."""
    import datasets
    recs, audio = datasets.load(cfg["datasets"].split(","))
    held = cfg["lodo"]
    if held.startswith("fold"):
        fold = datasets.child_folds(recs)
        is_test = lambda r: fold[r["pid"]] == int(held[4:])
    else:
        is_test = lambda r: r["dataset"] == held
    train = [r for r in recs if not is_test(r) and r["label"] is not None]
    print(f"{name}: training on {len(train)} recordings from {sorted({r['dataset'] for r in train})}", flush=True)
    models = []
    for s in (range(cfg["seed"], cfg["seed"] + cfg["seeds"]) if held != "none" else range(5)):
        m = train_fold(dict(cfg, seed=s), train, audio, dev)
        torch.save(m.state_dict(), out / f"fold{s}.pt")
        models.append(m)
    win = int(cfg["win_s"] * SR)
    row = dict(name=name, folds=f"left out: {held}", minutes=None)
    scores = {}
    if held != "none":
        test = [r for r in recs if is_test(r)]
        for cond in ("phone_v2", "clean"):
            for phone in (False, True):
                part = [r for r in test if r["phone"] == phone]
                if not part:
                    continue
                fn = None if cond == "clean" or phone else CONDITIONS[cond]
                p = np.mean(
                    [score(m, [audio[r["stem"]] for r in part], win, fn, dev, [r["spot"] for r in part])
                     for m in models],
                    axis=0)
                for r, v in zip(part, p):
                    scores.setdefault(r["stem"], {})[cond] = float(v)
            per = {}
            for d in sorted({r["dataset"] for r in test}):
                s, y = datasets.child_scores([r for r in test if r["dataset"] == d], scores, cond)
                f = s >= cutoff
                per[d] = dict(children=int(len(y)), murmur_children=int(y.sum()),
                              auroc=round(auroc(y, s), 4) if 0 < y.sum() < len(y) else None,
                              sensitivity=round(float(f[y].mean()), 4), specificity=round(float((~f[~y]).mean()), 4),
                              accuracy=round(float((f == y).mean()), 4))
            row[f"lodo_{cond}"] = per.get(held, per)
    json.dump(scores, open(out / "scores.json", "w"))
    _, _, our_paths, our_x = external_sets()
    if our_x:
        ours = dict(zip(
            [p.stem for p in our_paths],
            np.mean([score(m, our_x, win, None, dev) for m in models], axis=0).round(4).tolist()))
        json.dump(dict(ours=ours), open(out / "ours.json", "w"))
    row["minutes"] = round((time.time() - t_start) / 60, 1)
    row["settings"] = {k: v for k, v in cfg.items() if v != DEFAULTS[k]}
    with open(EXP / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def pascal_beyond_quality(y, score):
    """AUROC gain over recording quality alone on PASCAL (its murmurs are cleaner, so quality cheats)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    q = np.array(json.load(open(ROOT / "ml/out/pascal_quality.json"))["values"])
    p = np.clip(score, 1e-4, 1 - 1e-4)
    with_score = np.hstack([q, np.log(p / (1 - p))[:, None]])

    def loo(x):
        pred = np.zeros(len(y))
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x[keep], y[keep])
            pred[i] = clf.predict_proba(x[i:i + 1])[0, 1]
        return pred
    return auroc(y, loo(with_score)) - auroc(y, loo(q))


def log_result(name, out, folds, seconds, recs, oof, pascal, pascal_y, ours, settings, extra=None):
    json.dump(dict(oof=oof, pascal=np.round(pascal, 4).tolist(), ours=ours), open(out / "scores.json", "w"))
    scored = [r for r in recs if r["stem"] in oof]
    labeled = [r for r in scored if r["label"] is not None]
    conds = next(iter(oof.values())).keys()

    def mine(spot):
        return [s for stem, s in ours.items() if "_me_" in stem and stem.endswith(f"_{spot}")]

    def percentile(spot):  # where our median lands among CirCor normals at that spot
        normal = np.array([oof[r["stem"]]["phone_v2"] for r in scored if r["spot"] == spot and not r["child_murmur"]])
        return round(float((normal < np.median(mine(spot))).mean() * 100), 1) if mine(spot) else None

    row = dict(name=name, folds=folds, minutes=round(seconds / 60, 1))
    for cond in conds:
        row[f"cv_{cond}"] = round(child_auroc(scored, oof, cond, APP_SPOTS), 4)
    for cond in ("phone_v1", "phone_v2"):
        if cond in conds:
            row[f"cv_{cond}_allspots"] = round(child_auroc(scored, oof, cond), 4)
            row[f"cv_{cond}_recordings"] = round(
                auroc([r["label"] for r in labeled], [oof[r["stem"]][cond] for r in labeled]), 4)
    row["pascal"] = round(auroc(pascal_y, pascal), 4)
    row["pascal_beyond_quality"] = round(pascal_beyond_quality(pascal_y, np.asarray(pascal)), 4)
    # THIGH and AIR are no-heart checks
    row.update({
        f"me_{s}": round(float(np.mean(mine(s))), 3) if mine(s) else None
        for s in ("PV", "TV", "MV", "THIGH", "AIR")})
    if "phone_v2" in conds:
        row.update(me_PV_pct=percentile("PV"), me_TV_pct=percentile("TV"))
    row.update(extra or {})
    row["settings"] = settings
    with open(EXP / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], parse(sys.argv[2:]))
