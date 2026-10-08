from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from .config import load_config, setup_logging
from .db import Database
from .frigate_client import FrigateClient
from .geometry import is_plausible_box
from .vocab import effective_synonyms, label_in_text

logger = logging.getLogger(__name__)


def _select_pseudo_labels(cfg: dict, db: Database, class_map: dict) -> list:
    pseudo_cfg = cfg["training"].get("pseudo_labels", {}) or {}
    if not pseudo_cfg.get("enabled"):
        return []
    per_day = int(pseudo_cfg.get("per_camera_label_day", 3))
    max_per_class = int(pseudo_cfg.get("max_per_class", 250))
    synonyms = effective_synonyms(cfg, db)
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE status = 'confirmed' AND snapshot_path != '' ORDER BY start_time ASC"
        ).fetchall()
    picked = []
    per_class: dict = {}
    groups: dict = {}
    for row in rows:
        if not Path(row["snapshot_path"]).is_file():
            continue
        label = row["label"]
        if label not in class_map:
            continue
        if not is_plausible_box(row["box"]):
            continue
        if per_class.get(label, 0) >= max_per_class:
            continue
        description = row["description"] or ""
        if not label_in_text(label, description, synonyms):
            continue
        day = time.strftime("%Y%m%d", time.localtime(row["start_time"]))
        key = (row["camera"], label, day)
        if groups.get(key, 0) >= per_day:
            continue
        groups[key] = groups.get(key, 0) + 1
        per_class[label] = per_class.get(label, 0) + 1
        picked.append(row)
    return picked


def _region_crop(img, box, imgsz: int, min_side: int = 160):
    """Crop a square region the way Frigate feeds the detector at inference:
    ~1.35x the object's larger side, with black padding when the region extends
    past the frame (Frigate builds oversized regions for full-width detections).
    Training on the same domain as inference keeps the model from hallucinating
    on the content/padding boundary or on zoomed crops."""
    import cv2
    import numpy as np

    fh, fw = img.shape[:2]
    if box is None:
        side = random.choice([s for s in (320, 480, 640) if s <= min(fw, fh)] or [min(fw, fh)])
        side = max(side // 4 * 4, 4)
        x0 = random.randint(0, max(0, fw - side))
        y0 = random.randint(0, max(0, fh - side))
        crop = img[y0:y0 + side, x0:x0 + side]
        return cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), None

    px, py, pw, ph = box[0] * fw, box[1] * fh, box[2] * fw, box[3] * fh
    side = int(1.35 * max(pw, ph))
    side = max(side, min_side, 4)
    side = min(side, 2 * max(fw, fh))
    cx, cy = px + pw / 2.0, py + ph / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    dx0, dy0 = sx0 - x0, sy0 - y0
    cw = min(fw - sx0, side - dx0)
    ch = min(fh - sy0, side - dy0)
    if cw > 0 and ch > 0:
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = img[sy0:sy0 + ch, sx0:sx0 + cw]
    crop = cv2.resize(canvas, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    bx = min(max((px - x0) / side, 0.0), 1.0)
    by = min(max((py - y0) / side, 0.0), 1.0)
    bw = min(max(pw / side, 0.0), 1.0)
    bh = min(max(ph / side, 0.0), 1.0)
    return crop, (bx, by, bw, bh)


def export_dataset(cfg: dict, db: Database) -> dict:
    labels = list(cfg["labels"]["track"])
    class_map = {name: idx for idx, name in enumerate(labels)}
    dataset_dir = Path(cfg["training"]["dataset_dir"])
    val_split = float(cfg["training"]["val_split"])
    region_crops = bool(cfg["training"].get("region_crops", True))
    imgsz = int(cfg["training"]["imgsz"])

    all_rows = db.corrections_unexported()
    positives = []
    backgrounds = []
    skipped_ids = []
    for row in all_rows:
        if row["correct_label"] == "false_positive":
            if row["image_path"] and Path(row["image_path"]).is_file():
                backgrounds.append(row)
            else:
                skipped_ids.append(row["id"])
        elif (
            row["correct_label"] in class_map
            and row["image_path"]
            and Path(row["image_path"]).is_file()
            and row["box"]
        ):
            positives.append(row)
        else:
            skipped_ids.append(row["id"])
    rows = positives + backgrounds

    counts = {"exported": 0, "train": 0, "val": 0, "background": 0, "pseudo": 0}
    pseudo_rows = _select_pseudo_labels(cfg, db, class_map)

    # Drop pseudo-label files from previous exports so the dataset always
    # reflects the current selection (e.g. after an artifact filter changes).
    for sub in ("images/train", "labels/train"):
        stale_dir = dataset_dir / sub
        if stale_dir.is_dir():
            for stale in stale_dir.glob("pseudo_*"):
                stale.unlink(missing_ok=True)

    if not rows and not pseudo_rows:
        db.corrections_mark_exported(skipped_ids)
        if skipped_ids:
            logger.info("no exportable corrections, marked %d records as processed", len(skipped_ids))
        else:
            logger.info("no unexported corrections, dataset unchanged")
        return counts

    if len(rows) < 4:
        subsets = [("train", rows), ("val", list(rows))]
        logger.info("dataset is tiny (%d samples), using train == val", len(rows))
    else:
        random.shuffle(rows)
        split_index = int(len(rows) * (1.0 - val_split)) if val_split < 1 else len(rows) - 1
        split_index = max(1, min(split_index, len(rows) - 1))
        subsets = [
            ("train", rows[:split_index]),
            ("val", rows[split_index:]),
        ]

    exported_ids: list[int] = list(skipped_ids)

    for subset_name, subset_rows in subsets:
        images_dir = dataset_dir / "images" / subset_name
        labels_dir = dataset_dir / "labels" / subset_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for row in subset_rows:
            source = row["image_path"]
            target = images_dir / f"{row['event_id']}.jpg"
            annotation = labels_dir / f"{row['event_id']}.txt"
            for other in ("train", "val"):
                if other != subset_name:
                    (dataset_dir / "images" / other / f"{row['event_id']}.jpg").unlink(missing_ok=True)
                    (dataset_dir / "labels" / other / f"{row['event_id']}.txt").unlink(missing_ok=True)
            if row["correct_label"] == "false_positive":
                if region_crops:
                    try:
                        import cv2

                        img = cv2.imread(source)
                        if img is not None:
                            crop, _ = _region_crop(img, json.loads(row["box"]) if row["box"] else None, imgsz)
                            cv2.imwrite(str(target), crop)
                        else:
                            shutil.copyfile(source, target)
                    except Exception:
                        logger.exception("region crop failed for %s, falling back to full frame", row["event_id"])
                        shutil.copyfile(source, target)
                else:
                    shutil.copyfile(source, target)
                annotation.write_text("")
                counts["background"] += 1
            else:
                try:
                    x, y, w, h = json.loads(row["box"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning("skipping correction %s: bad box %r", row["id"], row["box"])
                    exported_ids.append(row["id"])
                    continue
                if w <= 0 or h <= 0:
                    logger.warning("skipping correction %s: degenerate box %r", row["id"], row["box"])
                    exported_ids.append(row["id"])
                    continue
                if region_crops:
                    try:
                        import cv2

                        img = cv2.imread(source)
                        if img is not None:
                            crop, box_c = _region_crop(img, (x, y, w, h), imgsz)
                            cv2.imwrite(str(target), crop)
                            x, y, w, h = box_c
                        else:
                            shutil.copyfile(source, target)
                    except Exception:
                        logger.exception("region crop failed for %s, falling back to full frame", row["event_id"])
                        shutil.copyfile(source, target)
                else:
                    shutil.copyfile(source, target)
                xc = min(max(x + w / 2.0, 0.0), 1.0)
                yc = min(max(y + h / 2.0, 0.0), 1.0)
                nw = min(max(w, 0.0), 1.0)
                nh = min(max(h, 0.0), 1.0)
                annotation.write_text(
                    f"{class_map[row['correct_label']]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
                )
            exported_ids.append(row["id"])
            counts["exported"] += 1
            counts[subset_name] += 1

    names_block = "\n".join(f"  {idx}: {name}" for name, idx in class_map.items())
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n"
    )
    (dataset_dir / "labels.txt").write_text("\n".join(labels) + "\n")

    if pseudo_rows:
        images_dir = dataset_dir / "images" / "train"
        labels_dir = dataset_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for row in pseudo_rows:
            source = row["snapshot_path"]
            try:
                x, y, w, h = json.loads(row["box"]) if row["box"] else (0, 0, 0, 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            xc, yc, nw, nh = x, y, w, h
            if region_crops:
                try:
                    import cv2

                    img = cv2.imread(source)
                    if img is not None:
                        crop, box_c = _region_crop(img, (x, y, w, h), imgsz)
                        cv2.imwrite(str(images_dir / f"pseudo_{row['id']}.jpg"), crop)
                        x, y, w, h = box_c
                    else:
                        shutil.copyfile(source, images_dir / f"pseudo_{row['id']}.jpg")
                except Exception:
                    logger.exception("region crop failed for pseudo %s, falling back to full frame", row["id"])
                    shutil.copyfile(source, images_dir / f"pseudo_{row['id']}.jpg")
            else:
                shutil.copyfile(source, images_dir / f"pseudo_{row['id']}.jpg")
            xc = min(max(x + w / 2.0, 0.0), 1.0)
            yc = min(max(y + h / 2.0, 0.0), 1.0)
            nw = min(max(w, 0.0), 1.0)
            nh = min(max(h, 0.0), 1.0)
            (labels_dir / f"pseudo_{row['id']}.txt").write_text(
                f"{class_map[row['label']]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
            )
            counts["pseudo"] += 1

    db.corrections_mark_exported(exported_ids)
    logger.info("exported %d corrections to %s (%s)", counts["exported"], dataset_dir, counts)
    return counts


def dataset_stats(cfg: dict) -> dict:
    """Summarize the on-disk YOLO dataset that the trainer actually consumes.

    Distinguishes human-labeled corrections from auto-generated pseudo labels and
    reports object counts per class for the train/val splits. Empty label files
    are background images (false-positive corrections), not objects.
    """
    dataset_dir = Path(cfg["training"]["dataset_dir"])
    labels_file = dataset_dir / "labels.txt"
    if labels_file.is_file():
        class_names = labels_file.read_text().split()
    else:
        class_names = list(cfg["labels"]["track"])

    result: dict = {
        "present": False,
        "dataset_dir": str(dataset_dir),
        "images": {"train": 0, "val": 0, "total": 0},
        "objects": {"train": 0, "val": 0, "total": 0},
        "backgrounds": {"train": 0, "val": 0, "total": 0},
        "human": {"images": 0, "objects": 0, "backgrounds": 0},
        "pseudo": {"images": 0, "objects": 0},
        "unknown_class_boxes": 0,
        "classes": [],
    }
    class_rows: dict = {}
    for split in ("train", "val"):
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.is_dir():
            continue
        result["present"] = True
        for path in sorted(labels_dir.glob("*.txt")):
            is_pseudo = path.name.startswith("pseudo_")
            source = "pseudo" if is_pseudo else "human"
            result["images"][split] += 1
            result["images"]["total"] += 1
            result[source]["images"] += 1
            try:
                lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            except OSError:
                continue
            if not lines:
                result["backgrounds"][split] += 1
                result["backgrounds"]["total"] += 1
                if not is_pseudo:
                    result["human"]["backgrounds"] += 1
                continue
            for line in lines:
                try:
                    class_idx = int(line.split()[0])
                except (ValueError, IndexError):
                    result["unknown_class_boxes"] += 1
                    continue
                label = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"#{class_idx}"
                row = class_rows.setdefault(
                    label,
                    {"label": label, "human": 0, "pseudo": 0, "train": 0, "val": 0, "total": 0},
                )
                row[source] += 1
                row[split] += 1
                row["total"] += 1
                result["objects"][split] += 1
                result["objects"]["total"] += 1
                result[source]["objects"] += 1
    result["classes"] = sorted(class_rows.values(), key=lambda r: (-r["total"], r["label"]))
    return result


def patch_model_path(config_text: str, model_path: str) -> tuple[str, bool]:
    lines = config_text.splitlines(keepends=True)
    in_model = False
    changed = False
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_model = stripped.rstrip(":").strip() == "model"
            continue
        if in_model and re.match(r"^\s*path\s*:", line):
            lines[i] = re.sub(r"path\s*:.*", f"path: {model_path}", line)
            changed = True
            break
    return "".join(lines), changed


def current_model_path(config_text: str) -> str:
    value = ""
    in_model = False
    for line in config_text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_model = stripped.rstrip(":").strip() == "model"
            continue
        if in_model:
            m = re.match(r"^\s*path\s*:\s*(.*)$", line)
            if m:
                value = m.group(1).strip().strip("'\"")
                break
    return value


def frigate_published_model_path(cfg: dict, name: str) -> str:
    prefix = str(cfg["frigate"].get("published_model_prefix", "/config/models/published")).rstrip("/")
    return f"{prefix}/{name}"


def list_published_models(cfg: dict) -> list[dict]:
    publish_dir = Path(cfg["training"]["publish_dir"]) / "published"
    models = []
    for f in sorted(publish_dir.glob("errata_*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True):
        models.append(
            {
                "name": f.name,
                "size_mb": round(f.stat().st_size / 1e6, 1),
                "created_at": f.stat().st_mtime,
            }
        )
    return models


def backup_frigate_config(cfg: dict, db: Database, reason: str) -> tuple[int, str]:
    client = FrigateClient(cfg["frigate"])
    text = client.get_config_text()
    active = current_model_path(text)
    backup_dir = Path(cfg["frigate"].get("backup_dir", "/data/frigate-config-backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = backup_dir / f"config-{stamp}-{reason}.yml"
    path.write_text(text)
    backup_id = db.frigate_backup_insert(time.time(), reason, active, str(path))
    keep = int(cfg["frigate"].get("backup_keep", 5))
    for stale_path in db.frigate_backups_prune(keep):
        if stale_path:
            Path(stale_path).unlink(missing_ok=True)
        logger.info("pruned old frigate config backup %s", stale_path)
    logger.info("backed up frigate config to %s (active model: %s)", path, active or "unknown")
    return backup_id, str(path)


def activate_model(cfg: dict, db: Database, model_name: str) -> dict:
    safe_name = Path(model_name).name
    if safe_name != model_name or not safe_name.endswith(".onnx"):
        return {"ok": False, "error": "invalid model name"}
    candidate = Path(cfg["training"]["publish_dir"]) / "published" / safe_name
    if not candidate.is_file():
        return {"ok": False, "error": f"model not found: {safe_name}"}
    backup_id, backup_path = backup_frigate_config(cfg, db, "pre-activate")
    client = FrigateClient(cfg["frigate"])
    current = client.get_config_text()
    frigate_path = frigate_published_model_path(cfg, safe_name)
    patched, changed = patch_model_path(current, frigate_path)
    if not changed:
        return {
            "ok": False,
            "error": "no top-level 'model: path:' block found in frigate config",
            "backup_id": backup_id,
        }
    resp = client.save_config(patched, "restart")
    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"frigate rejected config save (HTTP {resp.status_code}): {resp.text[:200]}",
            "backup_id": backup_id,
        }
    ready = client.wait_until_ready()
    db.kv_set("last_model_deployed", frigate_path)
    db.kv_set("last_model_deployed_at", str(time.time()))
    logger.info(
        "activated model %s in frigate (restart %s; backup %s)",
        frigate_path,
        "completed" if ready else "in progress",
        backup_path,
    )
    return {"ok": True, "model": frigate_path, "frigate_ready": ready, "backup_id": backup_id}


def restore_backup(cfg: dict, db: Database, backup_id: int) -> dict:
    row = db.frigate_backup_get(backup_id)
    if not row:
        return {"ok": False, "error": f"backup {backup_id} not found"}
    backup_file = Path(row["file_path"])
    if not backup_file.is_file():
        return {"ok": False, "error": f"backup file missing: {backup_file}"}
    backup_frigate_config(cfg, db, "pre-revert")
    client = FrigateClient(cfg["frigate"])
    resp = client.save_config(backup_file.read_text(), "restart")
    if resp.status_code != 200:
        return {"ok": False, "error": f"frigate rejected config save (HTTP {resp.status_code}): {resp.text[:200]}"}
    ready = client.wait_until_ready()
    logger.info("restored frigate config from %s (restart %s)", backup_file, "completed" if ready else "in progress")
    return {"ok": True, "restored_from": str(backup_file), "frigate_ready": ready}


def convert_onnx_input_to_nhwc(onnx_path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    if not graph.input:
        raise ValueError(f"{onnx_path} has no graph inputs")
    inp = graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(shape) != 4 or shape[1] != 3:
        logger.info("onnx input %s is not nchw (%s), skipping transpose", inp.name, shape)
        return onnx_path
    n, c, h, w = shape
    nchw_name = f"{inp.name}_nchw"
    graph.node.insert(
        0,
        helper.make_node(
            "Transpose",
            inputs=[inp.name],
            outputs=[nchw_name],
            perm=[0, 3, 1, 2],
            name=f"{inp.name}_to_nchw",
        ),
    )
    for node in graph.node[1:]:
        for i, ref in enumerate(node.input):
            if ref == inp.name:
                node.input[i] = nchw_name
    inp.CopyFrom(helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, [n, h, w, c]))
    onnx.checker.check_model(model)
    tmp = onnx_path.with_suffix(".nhwc.tmp")
    onnx.save(model, str(tmp))
    tmp.replace(onnx_path)
    logger.info("converted onnx input to nhwc %s -> [%s, %s, %s, %s]", shape, n, h, w, c)
    return onnx_path


def publish_model(cfg: dict, db: Database, onnx_path: Path, training_count: int, metrics: dict) -> Path:
    training = cfg["training"]
    publish_dir = Path(training["publish_dir"]) / "published"
    publish_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = publish_dir / f"errata_{stamp}.onnx"
    shutil.copyfile(onnx_path, dest)

    labels = list(cfg["labels"]["track"])
    (publish_dir / "labels.txt").write_text("\n".join(labels) + "\n")
    (publish_dir / "metrics.json").write_text(
        json.dumps({"created_at": time.time(), **metrics}, indent=2) + "\n"
    )
    db.model_version_insert(str(dest), training_count, json.dumps(metrics))

    keep = int(training["keep_versions"])
    versions = sorted(publish_dir.glob("errata_*.onnx"))
    for old in versions[:-keep] if keep > 0 else versions:
        logger.info("pruning old model version %s", old)
        old.unlink(missing_ok=True)
    return dest


def update_frigate_config(cfg: dict, db: Database, model_path: str) -> bool:
    client = FrigateClient(cfg["frigate"])
    try:
        current = client.get_config_text()
    except Exception:
        logger.exception("failed to fetch frigate config")
        return False
    patched, changed = patch_model_path(current, model_path)
    if not changed:
        logger.error("could not find model path in frigate config, aborting update")
        return False
    resp = client.save_config(patched)
    if resp.status_code != 200:
        logger.error("frigate rejected config save (HTTP %s): %s", resp.status_code, resp.text[:200])
        return False
    db.kv_set("last_model_created_at", str(time.time()))
    logger.info("frigate config updated and reload triggered for %s", model_path)
    return True


def run_training(
    cfg: dict,
    db: Database,
    device: str | None = None,
    on_progress=None,
) -> dict:
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    from ultralytics import YOLO

    training = cfg["training"]
    total_epochs = int(training["epochs"])
    requested_device = str(device) if device is not None else str(training.get("device", "cpu"))
    data_yaml = Path(training["dataset_dir"]) / "data.yaml"
    run_started = time.time()

    def emit(update: dict) -> None:
        if on_progress is None:
            return
        try:
            on_progress(update)
        except Exception:
            logger.debug("training progress callback failed", exc_info=True)

    def _train(device: str):
        model = YOLO(f"{training['model_type']}.pt")

        def _on_train_start(trainer):
            emit(
                {
                    "detail": "training",
                    "phase": "training",
                    "epoch": 0,
                    "epochs": int(getattr(trainer, "epochs", 0) or total_epochs),
                    "progress": 0.0,
                }
            )

        def _on_fit_epoch_end(trainer):
            epochs = int(getattr(trainer, "epochs", 0) or total_epochs)
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            metrics: dict = {}
            raw = getattr(trainer, "metrics", None) or {}
            for key, short in (
                ("metrics/mAP50(B)", "mAP50"),
                ("metrics/mAP50-95(B)", "mAP50-95"),
            ):
                value = raw.get(key)
                if value is not None:
                    try:
                        metrics[short] = round(float(value), 4)
                    except (TypeError, ValueError):
                        pass
            loss = getattr(trainer, "tloss", None)
            if loss is not None:
                try:
                    values = [round(float(v), 4) for v in list(loss)]
                except (TypeError, ValueError):
                    values = []
                if len(values) >= 3:
                    metrics.update(
                        {"box_loss": values[0], "cls_loss": values[1], "dfl_loss": values[2]}
                    )
            emit(
                {
                    "detail": "training",
                    "phase": "training",
                    "epoch": epoch,
                    "epochs": epochs,
                    "progress": round(epoch / epochs, 4) if epochs else 0.0,
                    "metrics": metrics,
                }
            )

        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        train_started = time.time()
        model.train(
            data=str(data_yaml),
            epochs=total_epochs,
            imgsz=int(training["imgsz"]),
            batch=int(training.get("batch", 16)) or 16,
            workers=int(training.get("workers", 8)),
            amp=bool(training.get("amp", False)),
            device=device,
            project=str(training["model_output_dir"]),
            name="train",
        )
        return model, time.time() - train_started

    try:
        model, train_seconds = _train(requested_device)
    except Exception:
        if requested_device == "cpu" or device is not None:
            raise
        logger.exception("training on device '%s' failed, falling back to cpu", requested_device)
        device = "cpu"
        model, train_seconds = _train(device)
    else:
        device = requested_device

    emit({"detail": "exporting onnx", "phase": "export"})
    best_weights = Path(model.trainer.best)
    onnx_path = Path(model.export(format="onnx", imgsz=int(training["imgsz"])))
    onnx_path = convert_onnx_input_to_nhwc(onnx_path)
    emit({"detail": "publishing model", "phase": "publish"})
    dataset = dataset_stats(cfg)
    duration_seconds = round(time.time() - run_started, 1)
    metrics = {
        "epochs": total_epochs,
        "imgsz": int(training["imgsz"]),
        "device": device,
        "duration_seconds": duration_seconds,
        "train_seconds": round(train_seconds, 1),
        "corrections_total": db.corrections_count(),
        "unexported_corrections": db.corrections_unexported_count(),
        "dataset": dataset,
    }
    dest = publish_model(cfg, db, onnx_path, db.corrections_count(), metrics)
    logger.info(
        "published model %s (weights %s; trained on %d images / %d objects: %s)",
        dest,
        best_weights,
        dataset["images"]["total"],
        dataset["objects"]["total"],
        {c["label"]: c["total"] for c in dataset["classes"]},
    )
    db.kv_set("last_model_created_at", str(time.time()))
    if training["auto_update_frigate_config"]:
        update_frigate_config(cfg, db, str(dest))
    return {"model": str(dest), "device": device, "dataset": dataset}


def refresh_snapshots(cfg: dict, db: Database) -> dict:
    client = FrigateClient(cfg["frigate"])
    snapshot_dir = Path(cfg["frigate"]["snapshot_dir"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stats = {"refreshed": 0, "recovered": 0, "missing": 0}

    with db.connect() as conn:
        events = conn.execute(
            "SELECT id, snapshot_path FROM events WHERE snapshot_path != ''"
        ).fetchall()
    for i, row in enumerate(events, 1):
        if client.download_snapshot(row["id"], row["snapshot_path"]):
            stats["refreshed"] += 1
        else:
            stats["missing"] += 1
        if i % 250 == 0:
            logger.info("refreshed %d/%d event snapshots", i, len(events))

    with db.connect() as conn:
        corrections = conn.execute(
            "SELECT id, event_id, image_path FROM corrections WHERE image_path != ''"
        ).fetchall()
    recovered = False
    for row in corrections:
        path = row["image_path"]
        if Path(path).is_file():
            continue
        if client.download_snapshot(row["event_id"], path):
            db.set_snapshot(row["event_id"], path)
            recovered = True
            stats["recovered"] += 1
        else:
            stats["missing"] += 1
    if recovered:
        db.corrections_reset_exported()

    logger.info(
        "snapshot refresh: %d refreshed, %d recovered, %d unavailable",
        stats["refreshed"],
        stats["recovered"],
        stats["missing"],
    )
    return stats


def degenerate_box_events(db: Database) -> list:
    with db.connect() as conn:
        events = conn.execute(
            "SELECT * FROM events WHERE box NOT IN ('', 'null')"
        ).fetchall()
    return [row for row in events if not is_plausible_box(row["box"])]


def log_box_report(rows: list) -> None:
    if not rows:
        logger.info("no events with degenerate detector boxes")
        return
    by_status = Counter(row["status"] for row in rows)
    logger.info(
        "degenerate detector boxes: %d event(s), by status %s", len(rows), dict(by_status)
    )
    by_camera = Counter((row["camera"], row["label"]) for row in rows)
    for (camera, label), n in by_camera.most_common():
        logger.info("  %s/%s: %d", camera, label, n)


def skip_degenerate_boxes(db: Database) -> None:
    rows = degenerate_box_events(db)
    log_box_report(rows)
    with db.connect() as conn:
        corrected = {
            r["event_id"] for r in conn.execute("SELECT DISTINCT event_id FROM corrections")
        }
    now = time.time()
    skipped = guarded = 0
    with db.connect() as conn:
        for row in rows:
            if row["status"] == "skipped":
                continue
            if row["id"] in corrected:
                guarded += 1
                continue
            conn.execute(
                "UPDATE events SET status = 'skipped', reviewed_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            skipped += 1
    logger.info(
        "marked %d event(s) skipped, %d left untouched (human-corrected), %d already skipped",
        skipped,
        guarded,
        len(rows) - skipped - guarded,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Errata trainer: export dataset and train")
    parser.add_argument("--config", default=None, help="path to errata config.yaml")
    parser.add_argument("--export-only", action="store_true", help="only export the YOLO dataset")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="wipe the dataset and re-export every correction from scratch (e.g. after an export format change)",
    )
    parser.add_argument(
        "--refresh-snapshots",
        action="store_true",
        help="re-download event snapshots without Frigate's bounding-box/label overlay",
    )
    parser.add_argument(
        "--check-boxes",
        action="store_true",
        help="report events with degenerate detector boxes (read-only)",
    )
    parser.add_argument(
        "--skip-degenerate-boxes",
        action="store_true",
        help="set status 'skipped' on events with degenerate boxes (human corrections are left untouched)",
    )
    parser.add_argument("--train", action="store_true", help="train a model (requires ultralytics)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    db = Database(cfg["database"]["path"])

    if args.check_boxes:
        rows = degenerate_box_events(db)
        log_box_report(rows)
        actionable = [r for r in rows if r["status"] in ("new", "flagged", "confirmed")]
        logger.info("%d event(s) not yet skipped; fix with --skip-degenerate-boxes", len(actionable))
        return

    if args.skip_degenerate_boxes:
        skip_degenerate_boxes(db)
        return

    if args.refresh_snapshots:
        refresh_snapshots(cfg, db)
        return

    if args.rebuild:
        dataset_dir = Path(cfg["training"]["dataset_dir"])
        for sub in ("images", "labels"):
            shutil.rmtree(dataset_dir / sub, ignore_errors=True)
        db.corrections_reset_exported()
        logger.info("dataset wiped and exported flags reset")

    if args.export_only or args.rebuild or not args.train:
        export_dataset(cfg, db)
        return

    if not db.corrections_count():
        logger.info("no corrections collected yet, nothing to train on")
        return
    export_dataset(cfg, db)
    run_training(cfg, db)


if __name__ == "__main__":
    main()
