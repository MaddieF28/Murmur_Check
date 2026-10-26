"""Runs each line of a queue file through experiments.py, skipping ones already done.
Usage: python ml/run_queue.py QUEUE_FILE
"""
import json
import subprocess
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
RESULTS = ML / "out/exp/results.jsonl"


def done():
    return {json.loads(line)["name"] for line in open(RESULTS) if line.strip()} if RESULTS.exists() else set()


def main():
    for line in open(sys.argv[1]):
        args = line.split("#")[0].split()
        # recheck done() every time in case something finished in the meantime
        if not args or args[0] in done():
            continue
        print(f"=== {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, str(ML / "experiments.py"), *args], check=False)


if __name__ == "__main__":
    main()
