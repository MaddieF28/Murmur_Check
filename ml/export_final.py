"""Export the all-data model (lodo=none) to one averaged ONNX file. NAME can join runs with "+".
Usage: .venv/bin/python ml/export_final.py NAME [--install]
"""
import json
import shutil
import sys
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch

import experiments as E
from crossval import Ensemble
from finalize import _ProbAsLogit
from lodo_report import run_folder
from model import SR
from train import single_file


def main():
    name = sys.argv[1]
    runs = [run_folder(n) for n in name.split("+")]
    cfgs = [dict(E.DEFAULTS, **json.load(open(f / "config.json"))) for f in runs]
    assert len({c["win_s"] for c in cfgs}) == 1, "every network in the app needs the same window length"
    nets = []
    for f, cfg in zip(runs, cfgs):
        for path in sorted(f.glob("fold*.pt")):
            m = E.build_model(cfg)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            nets.append(m.eval())
    folder = runs[0] if len(runs) == 1 else E.EXP / name.replace("+", "__")
    folder.mkdir(parents=True, exist_ok=True)
    ens, win = Ensemble(nets).eval(), int(cfgs[0]["win_s"] * SR)
    x = torch.randn(3, win) * 0.1  # just for checking the export
    onnx_path = folder / "ensemble.onnx"
    # export with 2 windows not 1 or the GRU gets stuck at batch size 1
    torch.onnx.export(ens, (x[:2],), onnx_path, input_names=["audio"], output_names=["prob"],
                      dynamic_axes={"audio": {0: "windows"}, "prob": {0: "windows"}})
    single_file(onnx_path)
    diff = float(np.abs(ort.InferenceSession(str(onnx_path)).run(None, {"audio": x.numpy()})[0]
                        - ens(x).detach().numpy()).max())
    assert diff < 1e-4, diff
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), matches PyTorch within {diff:.1e}")

    _, _, our_paths, our_x = E.external_sets()
    scores = E.score(_ProbAsLogit(ens), our_x, win, None, "cpu")
    by_spot = defaultdict(list)
    for p, s in zip(our_paths, scores):
        by_spot[p.stem.rsplit("_", 1)[-1]].append(float(s))
    json.dump(dict(zip([p.stem for p in our_paths], np.round(scores, 4).tolist())),
              open(folder / "ours.json", "w"), indent=2)
    print("our recordings (murmur probability):")
    for spot, v in sorted(by_spot.items()):
        print(f"  {spot:6} {len(v):3d} recordings, median {np.median(v):.3f}, range {min(v):.3f}-{max(v):.3f}")

    if "--install" in sys.argv:
        app = E.ROOT / "app/model"
        shutil.copy(onnx_path, app / "murmur.onnx")
        # lodo_report.py sets the cutoff, not this
        c = json.load(open(app / "config.json"))
        c.update(window=win, hop=win // 2)
        json.dump(c, open(app / "config.json", "w"), indent=2)
        print(f"installed into app/model ({len(nets)} networks, {win // SR} s windows)")


if __name__ == "__main__":
    main()
