from __future__ import annotations

import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIREMENTS_MAC = """ultralytics>=8.3
onnx>=1.16
onnxslim>=0.1
"""

TRAIN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: ./train-mac.sh

Environment overrides:
  EPOCHS=100         training epochs
  BATCH=16           batch size (keep small on CPU)
  IMGSZ=640          image size
  WORKERS=4          dataloader workers
  TRAIN_DEVICE=mps   mps on Apple Silicon (default), or cpu
  MODEL_TYPE=yolo11n base model
  PYTHON=python3     interpreter used to create the venv
USAGE
  exit 0
}

case "${1:-}" in
  -h|--help) usage ;;
esac

EPOCHS="${EPOCHS:-__EPOCHS__}"
IMGSZ="${IMGSZ:-__IMGSZ__}"
BATCH="${BATCH:-__BATCH__}"
WORKERS="${WORKERS:-__WORKERS__}"
TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
MODEL_TYPE="${MODEL_TYPE:-__MODEL_TYPE__}"
PYBIN="${PYTHON:-python3}"

if [ ! -f dataset/data.yaml ]; then
  echo "error: dataset/data.yaml missing - this kit contains no exported dataset yet" >&2
  echo "confirm some corrections in the Errata web UI and download a fresh kit" >&2
  exit 1
fi

if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "error: python3 not found - install it with: brew install python" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  "$PYBIN" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install -r requirements-mac.txt

export PYTORCH_ENABLE_MPS_FALLBACK=1
export YOLO_CONFIG_DIR="$(pwd)/.ultralytics"
exec .venv/bin/python - <<'PYEOF'
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

device = os.environ.get("TRAIN_DEVICE", "mps")
epochs = int(os.environ.get("EPOCHS", "100"))
imgsz = int(os.environ.get("IMGSZ", "640"))
batch = int(os.environ.get("BATCH", "16"))
workers = int(os.environ.get("WORKERS", "4"))
model_type = os.environ.get("MODEL_TYPE", "yolo11n")

import torch

if device == "mps" and not torch.backends.mps.is_available():
    print("mps is not available on this mac, training on cpu instead")
    device = "cpu"

ds = Path("dataset").resolve()
data_yaml = ds / "data.yaml"
meta = yaml.safe_load(data_yaml.read_text())
meta["path"] = str(ds)
data_yaml.write_text(yaml.safe_dump(meta, sort_keys=False))

def train(d):
    model = YOLO(f"{model_type}.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        amp=False,
        device=d,
        project="runs",
        name="train",
        exist_ok=True,
    )
    return model

try:
    model = train(device)
except Exception:
    if device == "cpu":
        raise
    print(f"training on '{device}' failed, falling back to cpu", file=sys.stderr)
    model = train("cpu")
    device = "cpu"

best = Path(model.trainer.best)
onnx = Path(model.export(format="onnx", imgsz=imgsz))

def to_nhwc(path):
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(str(path))
    g = m.graph
    if not g.input:
        raise ValueError(f"{path} has no graph inputs")
    inp = g.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(shape) != 4 or shape[1] != 3:
        print(f"onnx input {inp.name} is not nchw ({shape}), skipping transpose")
        return path
    n, c, h, w = shape
    nchw_name = inp.name + "_nchw"
    g.node.insert(0, helper.make_node(
        "Transpose", inputs=[inp.name], outputs=[nchw_name], perm=[0, 3, 1, 2], name=inp.name + "_to_nchw",
    ))
    for node in g.node[1:]:
        for i, ref in enumerate(node.input):
            if ref == inp.name:
                node.input[i] = nchw_name
    inp.CopyFrom(helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, [n, h, w, c]))
    onnx.checker.check_model(m)
    tmp = path.with_suffix(".nhwc.tmp")
    onnx.save(m, str(tmp))
    tmp.replace(path)
    print(f"converted onnx input to nhwc {shape} -> [{n}, {h}, {w}, {c}]")
    return path

onnx = to_nhwc(onnx)
stamp = time.strftime("%Y%m%d-%H%M%S")
out = Path("published")
out.mkdir(exist_ok=True)
dest = out / f"errata_{stamp}.onnx"
shutil.copyfile(onnx, dest)
if (ds / "labels.txt").is_file():
    shutil.copyfile(ds / "labels.txt", out / "labels.txt")
(out / "metrics.json").write_text(
    json.dumps({"created_at": time.time(), "epochs": epochs, "imgsz": imgsz, "device": device}, indent=2) + "\\n"
)
print()
print(f"model written to {dest}")
print(f"trained weights: {best}")
print()
print("copy the model back to the frigate host, e.g.:")
print("  scp published/errata_*.onnx published/labels.txt USER@FRIGATE-HOST:/path/to/errata-app/models/published/")
print("then point frigate's config at it (see README-MAC.md)")
PYEOF
"""

README_MD = """# Errata — Mac training kit

Retrain the Errata YOLO model on your Mac instead of the Frigate host. The
Train button on the server trains on the device configured in
`config.yaml` (`training.device`, with automatic CPU fallback); this kit is
an alternative that moves the heavy work to your Mac and brings the finished
model back to the server.

## What's inside

| Path | Purpose |
|---|---|
| `train-mac.sh` | One-shot trainer: creates a venv, installs deps, trains, exports ONNX |
| `requirements-mac.txt` | Python packages the trainer needs |
| `dataset/` | Your exported YOLO dataset (`images/`, `labels/`, `data.yaml`, `labels.txt`) |
| `errata/` | Errata source code (for reference, not needed to train) |
| `README-MAC.md` | This file |

## 1. Install dependencies (Homebrew)

Install Homebrew if you don't have it yet:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Follow the "Next steps" it prints so `brew` is on your PATH, then install
Python (3.10+):

    brew install python

That is all you need. `train-mac.sh` creates an isolated virtualenv (`.venv/`)
inside the kit folder and pip-installs everything else (torch, ultralytics,
onnx) on the first run.

Useful extras (optional):

    brew install git wget

Both Apple Silicon (M1–M4) and Intel Macs work. Training on CPU wants a few GB
of free RAM; close heavy apps for faster runs.

## 2. Train

    cd errata-mac-train
    chmod +x train-mac.sh
    ./train-mac.sh

On Apple Silicon this trains on the Metal GPU (MPS) automatically — no setup
needed. On Intel Macs it falls back to CPU. To force CPU on an Apple Silicon
Mac:

    TRAIN_DEVICE=cpu ./train-mac.sh

The first run installs torch (~2 GB of wheels) and downloads `yolo11n.pt`
(~6 MB). Later runs reuse `.venv/` and the Ultralytics cache.

Handy overrides:

    EPOCHS=50 BATCH=8 IMGSZ=480 ./train-mac.sh   # quick low-res run
    WORKERS=2 ./train-mac.sh                     # fewer dataloader workers

Keep `IMGSZ` at 640 for models you plan to deploy, since Frigate's model
config assumes 640.

### No Mac? Train on Google Colab

This kit also includes `train-colab.ipynb`. Open
[Google Colab](https://colab.research.google.com/), set the runtime to a free
**T4 GPU**, upload this zip, and run the notebook top to bottom — it installs
dependencies, trains, converts the ONNX input to NHWC, and downloads an
`errata-model.zip` containing `published/errata_<stamp>.onnx` + `labels.txt`.

## 3. Copy the model back to the Frigate host

The kit writes the trained model to `published/`:

    published/errata_YYYYMMDD-HHMMSS.onnx
    published/labels.txt
    published/metrics.json

Copy those to the Frigate host's model directory (scp, AirDrop, USB stick):

    scp published/errata_*.onnx published/labels.txt \\
        USER@FRIGATE-HOST:/path/to/errata-app/models/published/

Then point Frigate at the new model in `config/config.yml` on the host:

    model:
      path: /config/models/published/errata_YYYYMMDD-HHMMSS.onnx
      width: 640
      height: 640
      input_dtype: float
      labelmap_path: /config/models/published/labels.txt
      model_type: yolo-generic

Reload via the Frigate UI (Config → Save & Reload) or
`docker compose restart frigate`.

## Troubleshooting

- `error: dataset/data.yaml missing` — the kit had no exported dataset when it
  was downloaded. Confirm some corrections in the Errata web UI, then download
  a fresh kit.
- `python3: command not found` — run `brew install python`.
- MPS errors — force CPU with `TRAIN_DEVICE=cpu ./train-mac.sh`.
- Very slow training — that is CPU (Intel Mac); an Apple Silicon Mac uses MPS
  by default.
"""


def _zip_write_str(zf: zipfile.ZipFile, arcname: str, text: str, mode: int) -> None:
    info = zipfile.ZipInfo(arcname, date_time=time.localtime()[:6])
    info.external_attr = (mode & 0o777) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, text.encode("utf-8"))


def _render_script(training: dict, batch: int) -> str:
    script = TRAIN_SCRIPT
    script = script.replace("__EPOCHS__", str(int(training["epochs"])))
    script = script.replace("__IMGSZ__", str(int(training["imgsz"])))
    script = script.replace("__BATCH__", str(batch))
    script = script.replace("__WORKERS__", str(int(training.get("workers", 8))))
    script = script.replace("__MODEL_TYPE__", str(training["model_type"]))
    return script


def build_mac_kit(cfg: dict, db) -> Path:
    try:
        from .trainer import export_dataset

        export_dataset(cfg, db)
    except Exception:
        logger.exception("could not refresh the dataset before building the mac kit; bundling what exists")

    dataset_dir = Path(cfg["training"]["dataset_dir"])
    pkg_dir = Path(__file__).parent
    training = cfg["training"]
    batch = min(int(training.get("batch", 16) or 16), 16)

    fd, zip_path = tempfile.mkstemp(prefix="errata-mac-kit-", suffix=".zip")
    os.close(fd)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            root = "errata-mac-train"
            _zip_write_str(zf, f"{root}/README-MAC.md", README_MD, 0o644)
            _zip_write_str(zf, f"{root}/requirements-mac.txt", REQUIREMENTS_MAC, 0o644)
            _zip_write_str(zf, f"{root}/train-mac.sh", _render_script(training, batch), 0o755)
            colab_nb = pkg_dir / "train-colab.ipynb"
            if colab_nb.is_file():
                _zip_write_str(zf, f"{root}/train-colab.ipynb", colab_nb.read_text(), 0o644)
            for path in sorted(pkg_dir.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.name != "train-colab.ipynb"
                ):
                    zf.write(path, f"{root}/errata/{path.relative_to(pkg_dir).as_posix()}")
            if dataset_dir.is_dir():
                for path in sorted(dataset_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, f"{root}/dataset/{path.relative_to(dataset_dir).as_posix()}")
        return Path(zip_path)
    except Exception:
        try:
            os.unlink(zip_path)
        except OSError:
            pass
        raise
