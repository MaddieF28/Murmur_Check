"""One-time setup on the Windows PC (code, GPU packages, data), then starts the worker.
Run from an empty folder:   python setup_pc.py https://<the link>
"""
import io
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = sys.argv[1].rstrip("/")
HERE = Path.cwd()
VENV_PY = HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def get(url):
    return urllib.request.urlopen(url, timeout=300).read()


def step(msg):
    print(f"\n== {msg}", flush=True)


def main():
    step("Downloading the code")
    zipfile.ZipFile(io.BytesIO(get(f"{BASE}/pc/bundle.zip"))).extractall(HERE)

    step("Installing Python packages with GPU support (several minutes)")
    if not VENV_PY.exists():
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    pip = [str(VENV_PY), "-m", "pip", "install", "-q"]
    subprocess.run(pip + ["--upgrade", "pip"], check=True)
    # RTX 50-series needs a CUDA 12.8+ build, so try the newest first
    for cuda in ("cu130", "cu129", "cu128"):
        if subprocess.run(pip + ["torch", "--index-url", f"https://download.pytorch.org/whl/{cuda}"]).returncode == 0:
            print(f"installed PyTorch for {cuda}")
            break
    else:
        sys.exit("Couldn't install PyTorch with GPU support - check this window's text for the error.")
    subprocess.run(pip + ["numpy", "scipy", "scikit-learn", "onnx", "onnxscript", "onnxruntime",
                          "soundfile", "zstandard"],
                   check=True)

    step("Checking the graphics card")
    check = ("import torch; ok = torch.cuda.is_available(); "
             "print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); "
             "ok and print('GPU math works:', (torch.ones(2, device='cuda') * 2).sum().item() == 4)")
    if subprocess.run([str(VENV_PY), "-c", check]).returncode != 0:
        print("The GPU check failed. Updating the NVIDIA driver (NVIDIA app or GeForce Experience) usually fixes this.")

    step("Downloading datasets: CirCor, PASCAL A and B, ZCHSound (about 1.2 GB)")
    subprocess.run([str(VENV_PY), str(HERE / "ml/fetch_data.py"), "circor", "pascal", "pascalb", "zch"],
                   check=True)

    step("Setup done. Starting the worker - leave this window open and the PC awake.")
    print("(The worker also downloads a 6.6 GB outside test set once, before its first experiment.)")
    subprocess.run([str(VENV_PY), str(HERE / "pc/pc_worker.py"), BASE])


if __name__ == "__main__":
    main()
