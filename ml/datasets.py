"""Loads all the labeled heart-sound datasets into one format (list of recording dicts + audio by stem)."""
import csv
import io
import json
import math
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from data import load_circor, load_labels, load_phone
from model import SR

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("circor", "wvu", "wc2023", "ohh", "zch", "pascalb", "pascala")
SPOTTED = {"circor", "wvu", "wc2023", "ohh"}  # these have named chest spots
LOC = {"ULSB": "PV", "LLSB": "TV", "APEX": "MV", "URSB": "AV"}
COHORTS = {
    "wvu": ("WVU", "labels.json"),
    "wc2023": ("WC2023", "labels.json"),
    "ohh": ("OHH-2025", "labels_OHH-2025.json"),
}


def to_4k(x, sr):
    """Convert any audio to mono int16 at 4 kHz."""
    x = x.astype(np.float32) if x.ndim == 1 else x.astype(np.float32).mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return np.clip(x, -32768, 32767).astype(np.int16)


def fetch_ohh_from_mac():
    """For the PC: download the OHH-2025 files from the Mac (about 400 MB)."""
    base = ROOT / "pc/base_url.txt"
    if not base.exists():
        return
    url = base.read_text().strip() + "/pc/zenodo_ohh.zip"
    print(f"fetching OHH-2025 from {url}", flush=True)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=3600).read()))
    out = ROOT / "data/zenodo"
    for name in z.namelist():
        # skip anything with ".." so it can't write outside data/
        if name == "labels_OHH-2025.json" or (
                name.startswith("OHH-2025/") and name.endswith(".wav") and ".." not in name):
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_bytes(z.read(name))


def load(names, with_audio=True):
    """Recordings and audio for the named datasets (with_audio=False is labels only)."""
    recs, audio = [], {}

    def add(x, **r):
        recs.append(r)
        if with_audio:
            audio[r["stem"]] = x() if callable(x) else x

    # the r=r, m=m etc. defaults are so each lambda keeps its own loop value
    if "circor" in names:
        for r in load_labels():
            add(lambda r=r: load_circor(r["stem"]),
                stem=f"circor:{r['stem']}", pid=f"circor:{r['pid']}", dataset="circor",
                spot=r["spot"], label=r["label"], child_label=r["child_murmur"], phone=False)
    for key, (cohort, file) in COHORTS.items():
        if key not in names:
            continue
        path = ROOT / "data/zenodo" / file
        if key == "ohh" and not path.exists():
            fetch_ohh_from_mac()
        for m in json.load(open(path)).values():
            if m["project_name"] == cohort:
                y = int(m["has_murmur"])  # child-level label
                add(lambda m=m, cohort=cohort: wavfile.read(
                        ROOT / "data/zenodo" / cohort / f"{m['uuid']}.wav")[1],
                    stem=f"{key}:{m['uuid']}",
                    pid=f"{key}:{m['patient_id']}", dataset=key,
                    spot=LOC.get(str(m["recording_location"]).upper()),
                    label=y, child_label=bool(y), phone=False)
    if "zch" in names:  # VSD/PDA vs normal (ASD and PFO are often silent)
        import soundfile as sf
        base = ROOT / "data/zchsound/ZCHSound"
        labels = {"VSD": 1, "PDA": 1, "NORMAL": 0}
        for r in csv.DictReader(open(base / "Clean Heartsound Data Details.csv", encoding="utf-8-sig")):
            dx, f = r["diagnosis"].strip(), r["fileName"].strip()
            if dx in labels:
                # no child id, so each file is its own child
                add(lambda f=f: to_4k(*sf.read(base / "clean Heartsound Data" / f, dtype="int16")),
                    stem=f"zch:{f}", pid=f"zch:{f}", dataset="zch", spot="PV",
                    label=labels[dx], child_label=bool(labels[dx]), phone=False)
    if "pascalb" in names:
        import soundfile as sf
        for kind, y in (("murmur", 1), ("normal", 0)):
            for p in sorted((ROOT / "data/pascal_b" / kind).glob("*.wav")):
                add(lambda p=p: to_4k(*sf.read(p, dtype="int16")),
                    stem=f"pascalb:{p.stem}", pid=f"pascalb:{p.stem.split('_')[0]}", dataset="pascalb",
                    spot=None, label=y, child_label=bool(y), phone=False)
    if "pascala" in names:  # iPhone recordings that pass the app's checks
        for a in json.load(open(ROOT / "ml/out/pascal_accepted.json")):
            p = ROOT / a["path"]
            add(lambda p=p: np.clip(load_phone(p), -32768, 32767).astype(np.int16),
                stem=f"pascala:{p.parent.name}/{p.stem}",
                pid=f"pascala:{p.parent.name}/{p.stem}", dataset="pascala", spot=None, label=a["label"],
                child_label=bool(a["label"]), phone=True)
    return recs, audio


def child_folds(recs, k=5, seed=0):
    """Split children into k folds per dataset, murmur and no-murmur kids shuffled separately."""
    rng, groups, fold = random.Random(seed), {}, {}
    for r in recs:
        groups.setdefault((r["dataset"], r["child_label"]), set()).add(r["pid"])
    for key in sorted(groups):
        pids = sorted(groups[key])
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            fold[pid] = i % k
    return fold


def child_scores(test, scores, cond):
    """One score per child: their highest spot (only kids with all 3 spots) or highest recording."""
    spotted = test[0]["dataset"] in SPOTTED
    kids = {}
    for r in test:
        k = kids.setdefault(r["pid"], dict(y=bool(r["child_label"]), spots={}, s=-1.0))
        p = scores[r["stem"]][cond]
        if spotted:
            if r["spot"] in ("PV", "TV", "MV"):
                k["spots"][r["spot"]] = max(k["spots"].get(r["spot"], -1.0), p)
        else:
            k["s"] = max(k["s"], p)
    if spotted:
        kids = {
            pid: dict(k, s=max(k["spots"].values()))
            for pid, k in kids.items() if len(k["spots"]) == 3}
    return np.array([k["s"] for k in kids.values()]), np.array([k["y"] for k in kids.values()])
