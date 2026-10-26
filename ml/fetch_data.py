"""Downloads whichever datasets are missing (skips ones already done).
Usage: python ml/fetch_data.py circor pascal pascalb zch [zenodo]
"""
import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CIRCOR = "https://physionet-open.s3.amazonaws.com/circor-heart-sound/1.0.3"
PASCAL = "http://www.peterjbentley.com/heartchallenge/wav"
ZCH = "https://media.githubusercontent.com/media/WeiJieOvO/ZCHSound-Dataset/main/ZCHSound.zip"


def get(url):
    return urllib.request.urlopen(url, timeout=900).read()


def unzip_wavs(data, out):
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(data))
    for n in z.namelist():
        base = Path(n).name
        if base.endswith(".wav") and not base.startswith("._") and "__MACOSX" not in n:
            (out / base).write_bytes(z.read(n))


def circor():
    d = DATA / "circor"
    (d / "training_data").mkdir(parents=True, exist_ok=True)
    for name in ("training_data.csv", "RECORDS"):
        if not (d / name).exists():
            (d / name).write_bytes(get(f"{CIRCOR}/{name}"))
    stems = [s.strip() for s in (d / "RECORDS").read_text().splitlines() if s.strip()]
    pids = sorted({Path(s).name.split("_")[0] for s in stems})
    files = [f"{s}.wav" for s in stems] + [f"{s}.tsv" for s in stems] + [f"training_data/{p}.txt" for p in pids]

    def fetch(rel):
        if not (d / rel).exists():
            (d / rel).write_bytes(get(f"{CIRCOR}/{rel}"))

    # lots of tiny files, way faster in parallel
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, files))


def pascal():
    """Dataset A = real iPhone recordings"""
    for kind in ("normal", "murmur"):
        if not (DATA / "pascal" / kind).exists():
            unzip_wavs(get(f"{PASCAL}/Atraining_{kind}.zip"), DATA / "pascal" / kind)


def pascalb():
    for kind in ("normal", "murmur", "extrasystole"):
        if not (DATA / "pascal_b" / kind).exists():
            unzip_wavs(get(f"{PASCAL}/Btraining_{kind}.zip"), DATA / "pascal_b" / kind)


def zch():
    base = DATA / "zchsound"
    if not (base / "ZCHSound/Clean Heartsound Data Details.csv").exists():
        base.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(get(ZCH)))
        for n in z.namelist():
            if "__MACOSX" not in n and not Path(n).name.startswith("._"):
                z.extract(n, base)


def zenodo():
    if not (DATA / "zenodo/labels.json").exists():
        import zenodo_extract
        zenodo_extract.main()


STEPS = dict(circor=circor, pascal=pascal, pascalb=pascalb, zch=zch, zenodo=zenodo)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"== {name}", flush=True)
        STEPS[name]()
