from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .db import Database

logger = logging.getLogger(__name__)


def base_dir(cfg: dict) -> Path:
    return Path(str(cfg["training"].get("base_dir", "/data/base")))


def _yolo_dir(cfg: dict) -> Path:
    return base_dir(cfg) / "yolo"


def _frigate_plus_dir(cfg: dict) -> Path:
    return base_dir(cfg) / "frigate_plus"


def _uploads_dir(cfg: dict) -> Path:
    return base_dir(cfg) / "uploads"


def model_choices():
    from .trainer import MODEL_CHOICES

    return MODEL_CHOICES


def all_model_variants() -> list[str]:
    from .trainer import ALL_MODELS

    return list(ALL_MODELS)


def seed_builtin_models(db: Database) -> None:
    """Register the packaged YOLO variants as base models (idempotent)."""
    existing = {row["name"] for row in db.base_models()}
    for _group, variants in model_choices():
        for name in variants:
            if name in existing:
                continue
            db.base_model_insert(
                name=name, source="yolo", fmt="pt", path=None,
                imgsz=None, labels=None, trainable=True, meta=None,
            )


def onnx_input_size(path: str) -> tuple[int, int] | None:
    """Return (height, width) from an ONNX graph input, or None."""
    try:
        import onnx

        model = onnx.load(path)
        if not model.graph.input:
            return None
        dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        if len(dims) != 4:
            return None
        # Accept either NCHW (N,C,H,W) or NHWC (N,H,W,C).
        if dims[1] == 3:
            return int(dims[2]), int(dims[3])
        if dims[3] == 3:
            return int(dims[1]), int(dims[2])
    except Exception:
        logger.exception("could not read ONNX input shape from %s", path)
    return None


def list_base_models(db: Database) -> list[dict]:
    out = []
    for row in db.base_models():
        d = dict(row)
        d["exists"] = bool(d["path"]) and Path(d["path"]).is_file() if d["path"] else True
        out.append(d)
    return out


def import_frigate_plus(cfg: dict, db: Database, key: str, model_id) -> dict:
    from .plus_client import PlusClient, PlusError

    client = PlusClient(key)
    try:
        info = client.get_model_info(model_id)
        dest_dir = _frigate_plus_dir(cfg)
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = f"frigate_plus_{model_id}"
        dest = dest_dir / f"{name}.onnx"
        client.download_model(model_id, str(dest))
    except PlusError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("Frigate+ import failed")
        return {"ok": False, "error": str(exc)}

    (dest_dir / f"{name}.json").write_text(json.dumps(info, indent=2))
    size = onnx_input_size(str(dest))
    imgsz = size[0] if size else None
    labels = _write_plus_labels(dest_dir, name, info)
    db.base_model_insert(
        name=name, source="frigate_plus", fmt="onnx", path=str(dest),
        imgsz=imgsz, labels=labels, trainable=False,
        meta=json.dumps({"model_id": model_id, "name": info.get("name")}),
    )
    return {"ok": True, "name": name, "path": str(dest), "imgsz": imgsz, "info": info}


def _write_plus_labels(dest_dir: Path, name: str, info: dict) -> str | None:
    labels = info.get("labels") or info.get("labelmap")
    if not labels:
        return None
    if isinstance(labels, list):
        text = "\n".join(str(x) for x in labels) + "\n"
    else:
        text = str(labels)
    path = dest_dir / f"{name}.labels.txt"
    path.write_text(text)
    return str(path)


def store_upload(cfg: dict, db: Database, filename: str, data: bytes,
                 name: str | None = None) -> dict:
    safe = Path(filename).name
    suffix = Path(safe).suffix.lower()
    if suffix not in (".pt", ".onnx"):
        return {"ok": False, "error": "only .pt and .onnx base models are supported"}
    base_name = name or Path(safe).stem
    fmt = "pt" if suffix == ".pt" else "onnx"
    trainable = fmt == "pt"
    subdir = _uploads_dir(cfg)
    subdir.mkdir(parents=True, exist_ok=True)
    dest = subdir / safe
    dest.write_bytes(data)
    imgsz = None
    if fmt == "onnx":
        size = onnx_input_size(str(dest))
        if not size:
            dest.unlink(missing_ok=True)
            return {"ok": False, "error": "could not read ONNX input shape; not a valid model"}
        imgsz = size[0]
    db.base_model_insert(
        name=base_name, source="upload", fmt=fmt, path=str(dest),
        imgsz=imgsz, labels=None, trainable=trainable, meta=None,
    )
    return {"ok": True, "name": base_name, "format": fmt, "path": str(dest), "imgsz": imgsz}


def delete_base_model(db: Database, name: str) -> dict:
    row = db.base_model_get(name)
    if not row:
        return {"ok": False, "error": "base model not found"}
    if row["source"] in ("frigate_plus", "upload") and row["path"]:
        Path(row["path"]).unlink(missing_ok=True)
    db.base_model_delete(name)
    return {"ok": True, "name": name}
