"""Packs the code + small data files into capture/pc/ for the PC worker and bumps version.txt.
Run: .venv/bin/python pc/make_bundle.py
"""
import re
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "capture/pc"
FILES = ["ml/out/split.json", "ml/out/pascal_accepted.json", "ml/out/pascal_quality.json",
         "ml/out/phone_profile.npz"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("ml", "analysis", "pc"):
            for p in sorted((ROOT / folder).glob("*.py")):
                z.write(p, p.relative_to(ROOT).as_posix())
        # tunnel link changes every restart, so grab the newest one
        link = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com",
                          (ROOT / "capture/tunnel.log").read_text())[-1]
        z.writestr("pc/base_url.txt", link)
        for f in FILES:
            z.write(ROOT / f, f)
    shutil.copy(ROOT / "pc/setup_pc.py", OUT / "setup_pc.py")
    # send the outside cohorts so the PC doesn't need the whole 6.6 GB
    zen = ROOT / "data/zenodo"
    if (zen / "labels.json").exists() and not (OUT / "zenodo_small.zip").exists():
        with zipfile.ZipFile(OUT / "zenodo_small.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(zen / "labels.json", "labels.json")
            z.writestr("sent_to_mac.txt", "came from the Mac")
            for p in sorted(zen.glob("*/*.wav")):
                z.write(p, p.relative_to(zen).as_posix())
    (OUT / "version.txt").write_text(time.strftime("%Y%m%d-%H%M%S"))
    if not (OUT / "queue.txt").exists():
        (OUT / "queue.txt").write_text("")
    print(f"published {(OUT / 'bundle.zip').stat().st_size / 1e6:.1f} MB, "
          f"version {(OUT / 'version.txt').read_text()}")


if __name__ == "__main__":
    main()
