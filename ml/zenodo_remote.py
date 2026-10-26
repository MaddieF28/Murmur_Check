"""Pulls cohorts out of the Zenodo db over HTTP range requests instead of downloading all 6.6 GB.
Same output as zenodo_extract.py.
Usage: .venv/bin/python ml/zenodo_remote.py [schema] [cohort ...]   (default cohorts WVU WC2023)
"""
import http.client
import io
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

import apsw
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
URL = "https://zenodo.org/api/records/23460250/files/hestia_wvu_wc2023_physionet22_ohh2025.db/content"
SIZE = 6590955520
BLOCK = 1 << 16
READAHEAD = 16
OUT = ROOT / "data/zenodo"
COHORTS = tuple(a for a in sys.argv[1:] if a != "schema") or ("WVU", "WC2023")
LABELS = OUT / ("labels.json" if COHORTS == ("WVU", "WC2023") else f"labels_{'_'.join(COHORTS)}.json")


class RemoteFile:
    """Read-only file for sqlite that fetches blocks with HTTP range requests (LRU cached)."""

    stats = dict(requests=0, bytes=0)

    def __init__(self):
        u = urlparse(URL)
        self.host, self.path, self.conn = u.netloc, u.path, None
        self.cache = OrderedDict()
        self.last = -2  # to spot sequential reads

    def _get(self, start, end):
        for attempt in range(6):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.host, timeout=120)
                self.conn.request("GET", self.path, headers={"Range": f"bytes={start}-{end}", "Accept": "*/*",
                                                             "User-Agent": "curl/8.7.1"})
                r = self.conn.getresponse()
                data = r.read()
                if r.status in (301, 302, 303, 307, 308):
                    u = urlparse(r.getheader("Location"))
                    self.host, self.path, self.conn = u.netloc, u.path + (f"?{u.query}" if u.query else ""), None
                    continue
                if r.status != 206:  # 206 = partial content
                    raise IOError(f"status {r.status}")
                RemoteFile.stats["requests"] += 1
                RemoteFile.stats["bytes"] += len(data)
                return data
            except Exception as e:  # zenodo drops/throttles sometimes
                self.conn = None
                time.sleep(2 ** attempt)
                last = e
        raise IOError(f"range {start}-{end} failed: {last}")

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        n = READAHEAD if i == self.last + 1 else 1  # read ahead if sequential
        data = self._get(i * BLOCK, min(SIZE, (i + n) * BLOCK) - 1)
        for j in range(n):
            if data[j * BLOCK:(j + 1) * BLOCK]:
                self.cache[i + j] = data[j * BLOCK:(j + 1) * BLOCK]
        while len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return self.cache[i]

    def xRead(self, amount, offset):
        out = bytearray()
        while amount > 0 and offset < SIZE:
            b = self._block(offset // BLOCK)
            self.last = offset // BLOCK
            chunk = b[offset % BLOCK: offset % BLOCK + amount]
            out += chunk
            amount -= len(chunk)
            offset += len(chunk)
        return bytes(out)

    def xFileSize(self):
        return SIZE

    def xClose(self):
        pass

    def xLock(self, level):
        pass

    def xUnlock(self, level):
        pass

    def xCheckReservedLock(self):
        return False

    def xFileControl(self, op, ptr):
        return False

    def xSectorSize(self):
        return 4096

    def xDeviceCharacteristics(self):
        return 0

    def xSync(self, flags):
        pass

    def xTruncate(self, size):
        raise IOError("read-only")

    def xWrite(self, data, offset):
        raise IOError("read-only")


class RemoteVFS(apsw.VFS):

    def __init__(self):
        super().__init__("zenodo_http", "")

    def xOpen(self, name, flags):
        return RemoteFile()

    def xAccess(self, pathname, flags):
        return False

    def xFullPathname(self, name):
        return name


def decompress(blob):
    import zstandard
    return zstandard.ZstdDecompressor().decompress(blob, max_output_size=500_000_000)


def main():
    vfs = RemoteVFS()  # noqa: F841 (registered by construction)
    con = apsw.Connection("file:zenodo.db?immutable=1", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
                          vfs="zenodo_http")
    for name, sql in con.execute("SELECT name, sql FROM sqlite_master"):
        print(name, "|", sql)
    if sys.argv[1:] == ["schema"]:
        print(RemoteFile.stats)
        return

    cols = [r[1] for r in con.execute("PRAGMA table_info(audio_info)")]
    blob = [r[1] for r in con.execute("PRAGMA table_info(audio_raw_zstd)") if r[1] != "uuid"][0]
    rows = list(con.execute(f"SELECT {', '.join(cols)} FROM audio_info WHERE project_name IN ({','.join('?' * len(COHORTS))})",
                            COHORTS))
    print(f"{len(rows)} recordings in {COHORTS}; downloaded so far: {RemoteFile.stats}", flush=True)
    # go in rowid order so reads are mostly sequential through the file (way fewer requests)
    order = dict(con.execute(f"SELECT uuid, rowid FROM audio_info WHERE project_name IN ({','.join('?' * len(COHORTS))})",
                             COHORTS))
    rows.sort(key=lambda row: order[row[cols.index("uuid")]])
    labels = json.load(open(LABELS)) if LABELS.exists() else {}
    t0 = time.time()
    for i, row in enumerate(rows):
        meta = dict(zip(cols, row))
        out = OUT / str(meta["project_name"]) / f"{meta['uuid']}.wav"
        if not out.exists():
            (data,) = con.execute(f"SELECT {blob} FROM audio_raw_zstd WHERE uuid = ?", (meta["uuid"],)).fetchone()
            sr, x = wavfile.read(io.BytesIO(decompress(data)))
            x = x.astype(np.float32)
            if x.ndim > 1:
                x = x.mean(axis=1)
            g = math.gcd(4000, sr)
            y = resample_poly(x, 4000 // g, sr // g)
            if np.abs(x).max() <= 1.0:
                y = y * 32767
            out.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(out, 4000, np.clip(y, -32768, 32767).astype(np.int16))
        labels[str(meta["uuid"])] = {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                                     for k, v in meta.items()}
        if i % 100 == 0:
            json.dump(labels, open(LABELS, "w"))  # save progress so reruns can skip
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  {RemoteFile.stats['bytes'] / 1e6:.0f} MB in {RemoteFile.stats['requests']} requests, "
                  f"{time.time() - t0:.0f}s", flush=True)
    json.dump(labels, open(LABELS, "w"))
    print(f"wrote {len(labels)} recordings; downloaded {RemoteFile.stats['bytes'] / 1e6:.0f} MB in total")


if __name__ == "__main__":
    main()
