"""Downloads the 6.6 GB Zenodo db and writes the WVU, WC2023, OHH-2025 recordings as 4 kHz WAVs + labels.json.
Run on the PC (not enough disk on the Mac).
Usage: python ml/zenodo_extract.py
"""
import io
import json
import math
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/zenodo_db/hestia_wvu_wc2023_physionet22_ohh2025.db"
URL = "https://zenodo.org/api/records/23460250/files/hestia_wvu_wc2023_physionet22_ohh2025.db/content"
OUT = ROOT / "data/zenodo"
SR = 4000


def download():
    """Saves to .part first so a half-finished download doesn't look like the real file."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    part = DB.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=600) as r, open(part, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e9:.2f} of {total / 1e9:.2f} GB", end="", flush=True)
    part.rename(DB)
    print()


def decompress(blob):
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob, max_output_size=500_000_000)
    except ImportError:
        from compression import zstd  # Python 3.14+
        return zstd.decompress(blob)


def main():
    if not DB.exists():
        print("downloading the database (6.6 GB)")
        download()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    for table in ("audio_info", "audio_raw_zstd"):
        print(table, [c[1] for c in con.execute(f"PRAGMA table_info({table})")])
    cols = [c[1] for c in con.execute("PRAGMA table_info(audio_info)")]
    blob = [c[1] for c in con.execute("PRAGMA table_info(audio_raw_zstd)") if c[1] != "uuid"][0]
    print("example rows:", con.execute("SELECT * FROM audio_info WHERE project_name != 'physionet22' LIMIT 3").fetchall())

    labels = {}
    query = (f"SELECT {', '.join('a.' + c for c in cols)}, b.{blob} FROM audio_info a "
             f"JOIN audio_raw_zstd b ON a.uuid = b.uuid WHERE a.project_name != 'physionet22'")
    for i, row in enumerate(con.execute(query)):
        meta = dict(zip(cols, row[:-1]))
        sr, x = wavfile.read(io.BytesIO(decompress(row[-1])))
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        g = math.gcd(SR, sr)
        y = resample_poly(x, SR // g, sr // g)
        if np.abs(x).max() <= 1.0:  # float wavs
            y = y * 32767
        out = OUT / str(meta["project_name"]) / f"{meta['uuid']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(out, SR, np.clip(y, -32768, 32767).astype(np.int16))
        labels[str(meta["uuid"])] = {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                                     for k, v in meta.items()}
        if i % 500 == 0:
            print(f"  {i} recordings", flush=True)
    json.dump(labels, open(OUT / "labels.json", "w"))
    print(f"wrote {len(labels)} recordings to {OUT}")


if __name__ == "__main__":
    main()
