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


def onnx_input_layout(path: str) -> str | None:
    """Return 'nchw' or 'nhwc' for an ONNX graph input, or None."""
    try:
        import onnx

        model = onnx.load(path)
        if not model.graph.input:
            return None
        dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        if len(dims) != 4:
            return None
        if dims[1] == 3:
            return "nchw"
        if dims[3] == 3:
            return "nhwc"
    except Exception:
        logger.exception("could not read ONNX input layout from %s", path)
    return None


def _model_layout(row, path: str) -> str:
    meta = {}
    try:
        meta = json.loads(row["meta"]) if row["meta"] else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}
    declared = str(meta.get("inputShape") or "").lower()
    if declared in ("nchw", "nhwc"):
        return declared
    return onnx_input_layout(path) or "nhwc"


def activate_base_model(cfg: dict, db: Database, name: str) -> dict:
    """Deploy an ONNX base model to Frigate.

    Copies the ONNX and its labelmap into the Frigate-visible published dir,
    then patches the Frigate ``model:`` block (path/width/height/input_tensor/
    labelmap_path/input_dtype/model_type) and restarts Frigate. Trained-from
    base models (.pt) are train-only and cannot be deployed.
    """
    from .frigate_client import FrigateClient
    from .trainer import (
        backup_frigate_config,
        frigate_published_model_path,
        patch_model_config,
    )

    row = db.base_model_get(name)
    if not row:
        return {"ok": False, "error": f"base model not found: {name}"}
    if row["format"] != "onnx":
        return {"ok": False, "error": "only ONNX base models can be deployed (.pt is train-only)"}
    src = Path(row["path"]) if row["path"] else None
    if not src or not src.is_file():
        return {"ok": False, "error": "base model file is missing"}
    safe_name = Path(name).name
    if safe_name != name:
        return {"ok": False, "error": "invalid base model name"}

    labels_src = Path(row["labels"]) if row["labels"] else None
    if not labels_src or not labels_src.is_file():
        return {"ok": False, "error": "base model has no labelmap; cannot set labelmap_path"}

    publish_dir = Path(cfg["training"]["publish_dir"]) / "published"
    publish_dir.mkdir(parents=True, exist_ok=True)
    deploy_name = f"{safe_name}.onnx"
    label_name = f"{safe_name}.labels.txt"
    dest = publish_dir / deploy_name
    try:
        import shutil

        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        shutil.copyfile(labels_src, publish_dir / label_name)
    except OSError as exc:
        return {"ok": False, "error": f"could not stage model into published/: {exc}"}

    imgsz = row["imgsz"] or (onnx_input_size(str(dest)) or (None,))[0] or int(cfg["training"].get("imgsz", 320))
    layout = _model_layout(row, str(dest))

    backup_id, _ = backup_frigate_config(cfg, db, "pre-activate-base")
    client = FrigateClient(cfg["frigate"])
    current = client.get_config_text()
    frigate_path = frigate_published_model_path(cfg, deploy_name)
    frigate_labels = frigate_published_model_path(cfg, label_name)
    patched, changed = patch_model_config(
        current, frigate_path, width=imgsz, height=imgsz, input_tensor=layout,
        extra={"labelmap_path": frigate_labels, "input_dtype": "float", "model_type": "yolo-generic"},
    )
    if not changed:
        return {"ok": False, "error": "no top-level 'model:' block found in frigate config", "backup_id": backup_id}
    resp = client.save_config(patched, "restart")
    if resp.status_code != 200:
        return {"ok": False, "error": f"frigate rejected config save (HTTP {resp.status_code}): {resp.text[:200]}", "backup_id": backup_id}
    ready = client.wait_until_ready()
    db.kv_set("last_model_deployed", frigate_path)
    db.kv_set("last_model_deployed_at", str(time.time()))
    logger.info("activated base model %s -> %s (layout %s, imgsz %s)", name, frigate_path, layout, imgsz)
    return {"ok": True, "model": frigate_path, "layout": layout, "imgsz": imgsz,
            "frigate_ready": ready, "backup_id": backup_id}



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
    if imgsz is None:
        imgsz = _plus_declared_size(info)
    labels = _write_plus_labels(dest_dir, name, info)
    db.base_model_insert(
        name=name, source="frigate_plus", fmt="onnx", path=str(dest),
        imgsz=imgsz, labels=labels, trainable=False,
        meta=json.dumps({
            "model_id": model_id,
            "name": info.get("name"),
            "inputShape": info.get("inputShape"),
            "pixelFormat": info.get("pixelFormat"),
            "inputDataType": info.get("inputDataType"),
            "type": info.get("type"),
        }),
    )
    return {"ok": True, "name": name, "path": str(dest), "imgsz": imgsz,
            "labels": labels, "info": info}


def _plus_declared_size(info: dict) -> int | None:
    for key in ("width", "height"):
        value = info.get(key)
        try:
            if value:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _plus_label_list(info: dict) -> list[str] | None:
    labels = info.get("labelMap") or info.get("labelmap") or info.get("labels")
    if not labels:
        return None
    if isinstance(labels, dict):
        try:
            ordered = sorted(labels.items(), key=lambda kv: int(kv[0]))
            return [str(v) for _k, v in ordered]
        except (ValueError, TypeError):
            return [str(v) for _k, v in labels.items()]
    if isinstance(labels, list):
        return [str(x) for x in labels]
    return None


def _write_plus_labels(dest_dir: Path, name: str, info: dict) -> str | None:
    labels = _plus_label_list(info)
    if not labels:
        return None
    path = dest_dir / f"{name}.labels.txt"
    path.write_text("\n".join(labels) + "\n")
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
