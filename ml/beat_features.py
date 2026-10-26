"""Different kind of model: average all the heartbeats into one, then gradient boosting on that.
Usage: .venv/bin/python ml/beat_features.py NAME [copies=3] [condition=phone_v2]
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from crossval import assign_folds
from data import load_circor, load_labels, to_float
from experiments import CONDITIONS, EXP, ROOT, external_sets, log_result
from model import SR

FRAME, HOP = 256, 40  # 64 ms frames every 10 ms
EDGES = np.geomspace(25, 800, 9)
PHASES = 24


def band_energies(x):
    x = to_float(x)
    n = 1 + (len(x) - FRAME) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, FRAME), (x.strides[0] * HOP, x.strides[0])) * np.hanning(FRAME)
    p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    f = np.fft.rfftfreq(FRAME, 1 / SR)
    e = np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in zip(EDGES, EDGES[1:])]) + 1e-20
    e = np.log(e)
    return e - e.mean(1, keepdims=True)


def features(x):
    e = band_energies(x)
    env = e[:4].mean(0)
    env = np.minimum(env, np.percentile(env, 95))  # so one bump doesnt take over
    # heart period = lag where the loudness lines up best with itself
    z = env - env.mean()
    r = np.correlate(z, z, "full")[len(z) - 1:] / (z @ z + 1e-12)
    lags = np.arange(33, 151)  # 40-180 bpm
    period = int(lags[np.argmax(r[lags])])
    clarity = float(r[period])

    n_cycles = e.shape[1] // period
    cycles = e[:, :n_cycles * period].reshape(e.shape[0], n_cycles, period)
    idx = (np.arange(PHASES + 1) * period / PHASES).astype(int)
    folded = np.stack([cycles[:, :, a:b].mean(2) for a, b in zip(idx, idx[1:])], axis=2)
    beat = folded.mean(1)
    spread = folded.std(1).mean(1)

    # put "lub" first: of the 2 loudest spots it's the one before the shorter gap (systole is shorter)
    loud = beat[:4].mean(0)
    a = int(np.argmax(loud))
    masked = loud.copy()
    for d in range(-3, 4):
        masked[(a + d) % PHASES] = -np.inf
    b = int(np.argmax(masked))
    first = a if (b - a) % PHASES < (a - b) % PHASES else b
    beat = np.roll(beat, -first, axis=1)
    gap = (b - a) % PHASES if first == a else (a - b) % PHASES

    systole = beat[:, 2:max(3, gap - 1)].mean(1)
    diastole = beat[:, gap + 3:PHASES - 2].mean(1) if gap + 3 < PHASES - 2 else beat[:, -3:].mean(1)
    # harmonics don't care where the cycle starts, so these work even if lub is found wrong
    harm = np.abs(np.fft.rfft(beat, axis=1))[:, 1:7]
    harm = harm / (harm.sum(1, keepdims=True) + 1e-9)
    return np.concatenate([beat.ravel(), systole - diastole, spread, harm.ravel(),
                           [60 * 100 / period, clarity, gap / PHASES]])


N_BEAT = 8 * PHASES + 8


def select(f, keep):
    if keep == "all":
        return f
    harm = f[N_BEAT + 8:N_BEAT + 8 + 48]  # skip the 8 spread numbers
    return np.concatenate([f[:N_BEAT], harm])


def run(name, copies=3, condition="phone_v2", keep="all"):
    """One GBM per fold, logged the same way experiments.py does."""
    t0 = time.time()
    split = json.load(open(ROOT / "ml/out/split.json"))
    recs = [r for r in load_labels() if split[r["pid"]] != "test"]
    fold = assign_folds(recs)
    gen = torch.Generator().manual_seed(0)
    augment = CONDITIONS[condition] if condition != "clean" else None

    def featurize(x, spot, cond, jitter_copies=1):
        out = []
        for _ in range(jitter_copies):
            w = torch.from_numpy(to_float(x)[None])
            if cond == "train":
                from data import phone_v2
                w = phone_v2(w, gen, [spot])
            elif cond is not None:
                w = cond(w, gen, [spot])
            out.append(select(features(w[0].numpy()), keep))
        return out

    audio = {r["stem"]: load_circor(r["stem"]) for r in recs}
    print(f"{name}: featurizing", flush=True)
    train_x = {r["stem"]: featurize(audio[r["stem"]], r["spot"], "train" if condition != "clean" else None, copies)
               for r in recs if r["label"] is not None}
    eval_x = {c: {r["stem"]: featurize(audio[r["stem"]], r["spot"], fn)[0] for r in recs}
              for c, fn in CONDITIONS.items()}
    pascal_x, pascal_y, our_paths, our_x = external_sets()
    pascal_f = np.array([select(features(x), keep) for x in pascal_x])
    our_f = np.array([select(features(x), keep) for x in our_x])

    oof, pascal_scores, our_scores = {}, [], []
    for k in sorted(set(fold.values())):
        tr = [r for r in recs if fold[r["pid"]] != k and r["label"] is not None]
        x = np.concatenate([train_x[r["stem"]] for r in tr])
        y = np.repeat([r["label"] for r in tr], [len(train_x[r["stem"]]) for r in tr])
        # weight murmurs up since there are fewer of them
        w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=0).fit(x, y, sample_weight=w)
        held = [r for r in recs if fold[r["pid"]] == k]
        for c in CONDITIONS:
            for r, p in zip(held, clf.predict_proba(np.array([eval_x[c][r["stem"]] for r in held]))[:, 1]):
                oof.setdefault(r["stem"], {})[c] = float(p)
        pascal_scores.append(clf.predict_proba(pascal_f)[:, 1])
        our_scores.append(clf.predict_proba(our_f)[:, 1])
        print(f"  fold {k} done", flush=True)

    pascal = np.mean(pascal_scores, axis=0)
    ours = dict(zip([p.stem for p in our_paths], np.mean(our_scores, axis=0).round(4).tolist()))
    out = EXP / name
    out.mkdir(parents=True, exist_ok=True)
    log_result(name, out, "0,1,2,3,4", time.time() - t0, recs, oof, pascal, pascal_y, ours,
               dict(model="beat_features_gbm", copies=copies, train_condition=condition, keep=keep))


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    run(sys.argv[1], copies=int(kw.get("copies", 3)), condition=kw.get("condition", "phone_v2"),
        keep=kw.get("keep", "all"))
