"""PC worker: runs the experiments the Mac queues up and posts the results back.
Run: .venv\\Scripts\\python.exe pc\\pc_worker.py https://<the link>
"""
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = sys.argv[1].rstrip("/")
HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "ml/out/exp/results.jsonl"


def get(path):
    return urllib.request.urlopen(f"{BASE}/{path}", timeout=300).read()


def post(name, kind, body):
    req = urllib.request.Request(f"{BASE}/pc_result?name={name}&kind={kind}", data=body, method="POST")
    urllib.request.urlopen(req, timeout=300).read()


def prepare():
    """Fetch the datasets, then send the Mac the two small outside cohorts (once)."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile", "zstandard"])
    r = subprocess.run([sys.executable, str(HERE / "ml/fetch_data.py"),
                        "circor", "pascal", "pascalb", "zch", "zenodo"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    post("pc_status", "log",
         f"fetch_data exit {r.returncode}\n{r.stdout[-4000:]}\n--- errors ---\n{r.stderr[-4000:]}".encode())
    labels, marker = HERE / "data/zenodo/labels.json", HERE / "data/zenodo/sent_to_mac.txt"
    if not labels.exists() or marker.exists():
        return
    small = {u: m for u, m in json.load(open(labels)).items() if m["project_name"] in ("WVU", "WC2023")}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labels.json", json.dumps(small))
        for u, m in small.items():
            z.write(HERE / "data/zenodo" / m["project_name"] / f"{u}.wav", f"{m['project_name']}/{u}.wav")
    # 15 MB chunks, then a done message with the count
    data, size = buf.getvalue(), 15_000_000
    parts = (len(data) + size - 1) // size
    for i in range(parts):
        post("zenodo", f"zen_chunk{i}", data[i * size:(i + 1) * size])
    post("zenodo", f"zen_done{parts}", b"{}")
    marker.write_text("sent")
    print(f"sent the outside cohorts to the Mac ({len(data) / 1e6:.0f} MB)", flush=True)


def finished():
    if not RESULTS.exists():
        return {}
    return {json.loads(line)["name"]: line.strip() for line in open(RESULTS) if line.strip()}


SENT = HERE / "ml/out/exp/sent_to_mac.txt"


def send_unsent():
    """Send any finished experiments the Mac doesn't have yet (failed sends retry next loop)."""
    sent = set(SENT.read_text().split()) if SENT.exists() else set()
    for name, row in finished().items():
        if name in sent:
            continue
        out = HERE / "ml/out/exp" / name
        post(name, "row", row.encode())
        for kind in ("config", "scores"):
            post(name, kind, (out / f"{kind}.json").read_bytes())
        for pt in sorted(out.glob("fold*.pt")):
            post(name, pt.stem, pt.read_bytes())
        with open(SENT, "a") as f:
            f.write(name + "\n")
        print(f"sent results for {name}", flush=True)


def main():
    version, failed = None, set()
    me = Path(__file__).read_bytes()
    while True:
        try:
            v = get("pc/version.txt").decode().strip()
            if v != version:
                zipfile.ZipFile(io.BytesIO(get("pc/bundle.zip"))).extractall(HERE)
                if Path(__file__).read_bytes() != me:  # worker itself changed, restart
                    print("worker updated; restarting", flush=True)
                    os.execv(sys.executable, [sys.executable, __file__, BASE])
                version, failed = v, set()
                print(f"code version {v}", flush=True)
                prepare()
            send_unsent()
            queue = [line.split("#")[0].split() for line in get("pc/queue.txt").decode().splitlines()]
            todo = [a for a in queue if a and a[0] not in finished() and a[0] not in failed]
            if not todo:
                print("nothing to run; checking again in a minute", flush=True)
                time.sleep(60)
                continue
            args = todo[0]
            print(f"=== {' '.join(args)}", flush=True)
            subprocess.run([sys.executable, str(HERE / "ml/experiments.py"), *args])
            # no results line = it crashed, don't retry until new code
            if args[0] not in finished():
                failed.add(args[0])
                post(args[0], "row", json.dumps({"name": args[0], "failed": True}).encode())
            send_unsent()
        except Exception as e:
            print(f"problem: {e}; retrying in a minute", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
