from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from .db import Database
from .scheduler import Scheduler
from .vocab import SYNONYMS_OVERRIDE_KEY, effective_synonyms

logger = logging.getLogger(__name__)

REVIEW_STATUSES = [
    "pending",
    "new",
    "flagged",
    "corrected",
    "false_positive",
    "skipped",
    "confirmed",
    "ignored",
    "all",
]

FLAG_REASONS = ["mismatch", "low_confidence", "noise", "oversized"]

REVIEWED_STATUSES = {"corrected", "false_positive", "confirmed", "ignored", "skipped"}

VIEW_STATUSES = [
    ("pending", "Pending review"),
    ("corrected", "Corrected"),
    ("false_positive", "False positive"),
    ("confirmed", "Auto-confirmed"),
    ("ignored", "Ignored"),
    ("skipped", "Skipped"),
    ("all", "All events"),
]

BULK_STATUSES = ["skipped", "confirmed", "false_positive", "ignored"]
BULK_STATUS_ACTIONS = {
    "skipped": "skip",
    "confirmed": "confirm",
    "false_positive": "false_positive",
    "ignored": "ignore",
}


def format_ts(ts) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))


def format_duration(seconds) -> str:
    if seconds in (None, ""):
        return "—"
    try:
        seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    if seconds < 0:
        return "—"
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s" if sec else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def create_app(config: dict) -> FastAPI:
    db = Database(config["database"]["path"])
    scheduler = Scheduler(config, db)

    base = str(config["server"].get("base_path", "/errata")).strip()
    if base and not base.startswith("/"):
        base = "/" + base
    base = base.rstrip("/")
    labels = list(config["labels"]["track"])
    queue_limit = int(config["review"]["queue_limit"])

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["localts"] = format_ts
    templates.env.filters["humandur"] = format_duration
    app_version = os.environ.get("ERRATA_VERSION", "").strip() or "dev"
    templates.env.globals["app_version"] = app_version

    from .trainer import IMGSZ_CHOICES, MODEL_CHOICES

    templates.env.globals["model_choices"] = MODEL_CHOICES
    templates.env.globals["default_model_type"] = str(config["training"].get("model_type", "yolov9s"))
    templates.env.globals["imgsz_choices"] = IMGSZ_CHOICES
    templates.env.globals["default_imgsz"] = int(config["training"].get("imgsz", 320))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        scheduler.start()
        yield
        scheduler.stop()

    app = FastAPI(title="Errata", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    router = APIRouter(prefix=base)

    def render(request: Request, template: str, ctx: dict, status_code: int = 200):
        ctx.update({"request": request, "base": base})
        return templates.TemplateResponse(request, template, ctx, status_code=status_code)

    @router.get("/", response_class=HTMLResponse)
    def dashboard(
        request: Request,
        camera: str = "",
        label: str = "",
        reason: str = "",
        corrected: str = "",
        status: str = "pending",
        msg: str = "",
    ):
        purged = db.purge_missing_snapshots()
        if purged and not msg:
            msg = f"Removed {purged} event(s) whose snapshot file no longer exists."
        rows = db.queue(
            status=status or "pending",
            camera=camera or None,
            label=label or None,
            reason=reason or None,
            corrected=corrected or None,
            limit=queue_limit,
        )
        ctx = {
            "rows": rows,
            "labels": labels,
            "cameras": db.distinct_cameras(),
            "event_labels": db.distinct_labels(),
            "corrected_labels": db.corrected_labels(),
            "reasons": FLAG_REASONS,
            "statuses": REVIEW_STATUSES,
            "view_statuses": VIEW_STATUSES,
            "reviewed_statuses": REVIEWED_STATUSES,
            "bulk_statuses": BULK_STATUSES,
            "filters": {
                "camera": camera,
                "label": label,
                "reason": reason,
                "corrected": corrected,
                "status": status or "pending",
            },
            "pending_count": db.queue_count("pending"),
            "total_count": db.queue_count("all"),
            "msg": msg,
        }
        return render(request, "index.html", ctx)

    @router.get("/summary", response_class=HTMLResponse)
    def summary(request: Request):
        from .trainer import dataset_stats

        corrections_since_epoch = db.kv_get("last_model_created_at")
        since = float(corrections_since_epoch) if corrections_since_epoch else 0
        threshold = int(config["training"]["trigger_threshold"])
        versions = db.model_versions(limit=5)
        latest = versions[0] if versions else None
        version_rows = []
        for v in versions:
            row = dict(v)
            try:
                metrics = json.loads(v["metrics"]) if v["metrics"] else {}
            except (json.JSONDecodeError, TypeError):
                metrics = {}
            row["dataset"] = metrics.get("dataset")
            row["duration_seconds"] = metrics.get("duration_seconds")
            row["train_seconds"] = metrics.get("train_seconds")
            row["model_type"] = metrics.get("model_type")
            version_rows.append(row)
        ctx = {
            "counts": db.counts_by_status(),
            "corrections_by_camera": db.corrections_by_group("camera"),
            "corrections_by_label": db.corrections_by_group("correct_label"),
            "corrections_total": db.corrections_count(),
            "corrections_since_model": db.corrections_by_group("correct_label", since),
            "threshold": threshold,
            "since_count": sum(int(r["n"]) for r in db.corrections_by_group("correct_label", since)),
            "latest_model": latest,
            "versions": version_rows,
            "dataset": dataset_stats(config),
            "unexported": db.corrections_unexported_count(),
            "last_run": scheduler.last_run,
            "last_stats": scheduler.last_stats,
            "train_state": train_state,
            "msg": request.query_params.get("msg", ""),
        }
        return render(request, "summary.html", ctx)

    @router.get("/synonyms", response_class=HTMLResponse)
    def synonyms_page(request: Request, msg: str = ""):
        synonyms = effective_synonyms(config, db)
        ctx = {
            "synonyms": synonyms,
            "labels": labels,
            "extra_labels": [k for k in synonyms if k not in labels],
            "msg": msg,
        }
        return render(request, "synonyms.html", ctx)

    @router.post("/synonyms")
    async def synonyms_save(request: Request):
        form = await request.form()
        override: dict[str, list[str]] = {}
        for key, value in form.multi_items():
            if not key.startswith("sym_"):
                continue
            label = key[4:]
            words = [w.strip() for w in str(value).split(",") if w.strip()]
            if words:
                override[label] = words
        db.kv_set(SYNONYMS_OVERRIDE_KEY, json.dumps(override))
        return RedirectResponse(
            f"{base}/synonyms?msg={quote('Synonyms saved. Applied on the next analyzer run.')}",
            status_code=303,
        )

    @router.post("/synonyms/reset")
    def synonyms_reset():
        db.kv_delete(SYNONYMS_OVERRIDE_KEY)
        return RedirectResponse(
            f"{base}/synonyms?msg={quote('Synonyms reset to the config.yaml baseline.')}",
            status_code=303,
        )

    def _parse_box_form(raw: str) -> str | None:
        """Validate a client-supplied box (JSON [x, y, w, h], frame-relative)."""
        if not raw or not raw.strip():
            return None
        try:
            x, y, w, h = (float(v) for v in json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid box: expected JSON [x, y, w, h]")
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0 - x)
        h = min(max(h, 0.0), 1.0 - y)
        if w < 0.01 or h < 0.01:
            raise HTTPException(status_code=400, detail="box too small after clamping")
        return json.dumps([round(x, 6), round(y, 6), round(w, 6), round(h, 6)])

    def apply_review(row, action: str, label: str = "", box_override: str | None = None) -> str:
        event_id = row["id"]
        if action == "confirm":
            new_label = label.strip()
            if not new_label or new_label == "false_positive":
                raise HTTPException(status_code=400, detail="no label provided")
            db.delete_corrections_for_event(event_id)
            db.insert_correction(
                {
                    "event_id": event_id,
                    "image_path": row["snapshot_path"] or "",
                    "correct_label": new_label,
                    "original_label": row["label"],
                    "confidence": row["confidence"],
                    "camera": row["camera"],
                    "box": box_override or row["box"],
                }
            )
            db.set_status(event_id, "corrected", reviewed=True)
            return f"{event_id} confirmed as '{new_label}' and queued for training"
        if action == "false_positive":
            db.delete_corrections_for_event(event_id)
            db.insert_correction(
                {
                    "event_id": event_id,
                    "image_path": row["snapshot_path"] or "",
                    "correct_label": "false_positive",
                    "original_label": row["label"],
                    "confidence": row["confidence"],
                    "camera": row["camera"],
                    "box": box_override or row["box"],
                }
            )
            db.set_status(event_id, "false_positive", reviewed=True)
            return f"{event_id} marked as false positive"
        if action == "skip":
            db.delete_corrections_for_event(event_id)
            db.set_status(event_id, "skipped", reviewed=True)
            return f"{event_id} skipped (excluded from training)"
        if action == "ignore":
            db.delete_corrections_for_event(event_id)
            db.set_status(event_id, "ignored", reviewed=True)
            return f"{event_id} ignored"
        if action == "reset":
            db.reset_event(event_id)
            return f"{event_id} review undone and returned to the queue"
        raise HTTPException(status_code=400, detail=f"unknown action '{action}'")

    @router.post("/review/{event_id}")
    def review(
        event_id: str,
        action: str = Form(...),
        label: str = Form(""),
        box: str = Form(""),
        return_to: str = Form(""),
    ):
        row = db.get_event(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        snapshot = row["snapshot_path"]
        if not snapshot or not Path(snapshot).is_file():
            db.delete_event(event_id)
            params = dict(parse_qsl(return_to, keep_blank_values=True))
            params.pop("msg", None)
            params["msg"] = f"{event_id} removed: its snapshot file is missing"
            return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)
        msg = apply_review(row, action, label, _parse_box_form(box))
        params = dict(parse_qsl(return_to, keep_blank_values=True))
        params.pop("msg", None)
        params["msg"] = msg
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    @router.post("/bulk")
    def bulk(
        camera: str = Form(""),
        label: str = Form(""),
        reason: str = Form(""),
        corrected: str = Form(""),
        status: str = Form(...),
        mode: str = Form("status"),
        new_label: str = Form(""),
        return_to: str = Form(""),
    ):
        params = dict(parse_qsl(return_to, keep_blank_values=True))
        params.pop("msg", None)
        if mode == "label":
            new_label = new_label.strip()
            if not new_label or new_label == "false_positive":
                raise HTTPException(status_code=400, detail="no label provided")
            if new_label not in labels:
                raise HTTPException(status_code=400, detail=f"unknown label '{new_label}'")
            rows = db.queue(
                status=status or "pending",
                camera=camera or None,
                label=label or None,
                reason=reason or None,
                corrected=corrected or None,
                limit=5000,
            )
            for row in rows:
                apply_review(row, "confirm", new_label)
            params["msg"] = (
                f"Bulk set label '{new_label}' on {len(rows)} event(s) matching "
                f"the current view filters"
            )
            return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)
        action = BULK_STATUS_ACTIONS.get(status)
        if action is None:
            raise HTTPException(status_code=400, detail=f"cannot bulk-apply status '{status}'")
        rows = db.pending_matching(camera or None, label or None, reason or None)
        for row in rows:
            apply_review(row, action, row["label"] if action == "confirm" else "")
        scope = ", ".join(
            filter(
                None,
                [
                    f"camera={camera}" if camera else "",
                    f"label={label}" if label else "",
                    f"reason={reason}" if reason else "",
                ],
            )
        ) or "all pending"
        params["msg"] = f"Bulk set '{status}' on {len(rows)} event(s) matching {scope}"
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    def _annotate(path: str, box: str, label: str) -> bytes | None:
        try:
            from PIL import Image, ImageDraw

            img = Image.open(path).convert("RGB")
            w_px, h_px = img.size
            x, y, w, h = json.loads(box)
            x0 = max(0, int(x * w_px))
            y0 = max(0, int(y * h_px))
            x1 = min(w_px - 1, int((x + w) * w_px))
            y1 = min(h_px - 1, int((y + h) * h_px))
            if x1 <= x0 or y1 <= y0:
                return None
            draw = ImageDraw.Draw(img)
            color = (0, 220, 255)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, w_px // 400))
            text = label or ""
            if text:
                ty = max(0, y0 - 16)
                draw.rectangle([x0, ty, x0 + 8 * len(text) + 8, ty + 15], fill=(0, 0, 0))
                draw.text((x0 + 4, ty + 2), text, fill=color)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception:
            logger.exception("failed to annotate snapshot %s", path)
            return None

    def _snapshot_or_404(event_id: str) -> str:
        row = db.get_event(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        path = row["snapshot_path"]
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=404, detail="snapshot not available")
        return path

    @router.get("/api/snapshots/{event_id}.jpg")
    def snapshot(event_id: str):
        path = _snapshot_or_404(event_id)
        row = db.get_event(event_id)
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        # Prefer the box from the latest correction so an edited box is what the
        # card shows; false-positive corrections have no box to draw.
        box = row["box"] if row else ""
        label = row["label"] if row else ""
        corr = db.latest_correction(event_id)
        if corr:
            if corr["correct_label"] == "false_positive":
                box = ""
            elif corr["box"]:
                box = corr["box"]
                label = corr["correct_label"] or label
        if box:
            annotated = _annotate(path, box, label or "")
            if annotated is not None:
                return Response(content=annotated, media_type="image/jpeg", headers=no_cache)
        return FileResponse(path, media_type="image/jpeg", headers=no_cache)

    @router.get("/api/snapshots/{event_id}/clean.jpg")
    def snapshot_clean(event_id: str):
        """Snapshot without the burned-in detection box, for the box editor."""
        path = _snapshot_or_404(event_id)
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        return FileResponse(path, media_type="image/jpeg", headers=no_cache)

    @router.get("/api/queue.json")
    def queue_json(status: str = "pending"):
        rows = db.queue(status=status, limit=queue_limit)
        return JSONResponse(
            {
                "generated_at": time.time(),
                "status": status,
                "count": len(rows),
                "events": [dict(r) for r in rows],
            }
        )

    @router.get("/api/health")
    def health():
        return {
            "ok": True,
            "version": app_version,
            "time": time.time(),
            "last_scheduler_run": scheduler.last_run,
            "last_scheduler_stats": scheduler.last_stats,
            "pending": db.queue_count("pending"),
            "corrections": db.corrections_count(),
        }

    train_state = {
        "running": False,
        "started_at": None,
        "finished_at": None,
        "ok": None,
        "detail": "no training run yet",
        "phase": None,
        "model_type": None,
        "imgsz": None,
        "epoch": None,
        "epochs": None,
        "progress": None,
        "metrics": None,
    }

    def _train_worker(model_type: str | None = None, imgsz: int | None = None) -> None:
        chosen = model_type or str(config["training"]["model_type"])
        chosen_imgsz = int(imgsz or config["training"]["imgsz"])
        train_state.update(
            {
                "running": True,
                "started_at": time.time(),
                "finished_at": None,
                "ok": None,
                "detail": "training",
                "phase": "starting",
                "model_type": chosen,
                "imgsz": chosen_imgsz,
                "epoch": None,
                "epochs": None,
                "progress": None,
                "metrics": None,
            }
        )
        try:
            from .trainer import export_dataset, run_training

            train_state["detail"] = "exporting dataset"
            train_state["phase"] = "exporting dataset"
            exported = export_dataset(config, db, imgsz=imgsz)
            train_state["detail"] = "training"
            result = run_training(
                config,
                db,
                device=None,
                on_progress=train_state.update,
                model_type=model_type,
                imgsz=imgsz,
            )
            result["exported"] = exported
            db.kv_set("last_train_at", str(time.time()))
            train_state.update(
                {
                    "running": False,
                    "finished_at": time.time(),
                    "ok": True,
                    "detail": json.dumps(result),
                    "phase": "done",
                    "progress": 1.0,
                }
            )
        except Exception as exc:
            logger.exception("training job failed")
            train_state.update(
                {
                    "running": False,
                    "finished_at": time.time(),
                    "ok": False,
                    "detail": str(exc),
                    "phase": "failed",
                }
            )

    @router.post("/api/train")
    def start_train(model_type: str = "", imgsz: int = 0):
        from .trainer import ALL_MODELS, IMGSZ_CHOICES

        model_type = model_type.strip()
        if model_type and model_type not in ALL_MODELS:
            return JSONResponse({"ok": False, "detail": f"unknown model '{model_type}'"}, status_code=400)
        if imgsz and imgsz not in IMGSZ_CHOICES:
            return JSONResponse(
                {"ok": False, "detail": f"unsupported imgsz '{imgsz}'; choose {IMGSZ_CHOICES}"},
                status_code=400,
            )
        if not config["training"].get("enabled", True):
            return JSONResponse(
                {"ok": False, "detail": "training is disabled in this container; download the Mac kit or use deploy/train.sh"},
                status_code=403,
            )
        if train_state["running"]:
            return JSONResponse({"ok": False, "detail": "a training run is already in progress"}, status_code=409)
        if importlib.util.find_spec("ultralytics") is None:
            return JSONResponse(
                {"ok": False, "detail": "ultralytics is not installed in this image; use the Mac kit or deploy/train.sh"},
                status_code=503,
            )
        threading.Thread(
            target=_train_worker,
            args=(model_type or None, imgsz or None),
            name="errata-train",
            daemon=True,
        ).start()
        return JSONResponse(
            {
                "ok": True,
                "detail": "training started",
                "model_type": model_type or config["training"]["model_type"],
                "imgsz": int(imgsz or config["training"]["imgsz"]),
            },
            status_code=202,
        )

    deploy_lock = threading.Lock()

    @router.get("/api/dataset/stats")
    def api_dataset_stats():
        from .trainer import dataset_stats

        stats = dataset_stats(config)
        stats["corrections_total"] = db.corrections_count()
        stats["unexported_corrections"] = db.corrections_unexported_count()
        return JSONResponse(stats)

    @router.get("/api/models")
    def api_models():
        from .frigate_client import FrigateClient
        from .trainer import current_model_path, list_published_models

        active = ""
        error = None
        try:
            active = current_model_path(FrigateClient(config["frigate"]).get_config_text())
        except Exception as exc:
            error = str(exc)
        versions = {Path(v["model_path"]).name: v for v in db.model_versions(limit=50)}
        models = list_published_models(config)
        for m in models:
            v = versions.get(m["name"])
            if v:
                m["training_count"] = v["training_count"]
                try:
                    metrics = json.loads(v["metrics"]) if v["metrics"] else {}
                except (json.JSONDecodeError, TypeError):
                    metrics = {}
                m["training_dataset"] = metrics.get("dataset")
                m["duration_seconds"] = metrics.get("duration_seconds")
                m["train_seconds"] = metrics.get("train_seconds")
                m["model_type"] = metrics.get("model_type")
        return JSONResponse(
            {
                "active": active,
                "active_name": Path(active).name if active else "",
                "models": models,
                "backups": [dict(b) for b in db.frigate_backups()],
                "error": error,
            }
        )

    @router.post("/api/models/activate")
    async def api_model_activate(request: Request):
        from .trainer import activate_model

        body = await request.json()
        name = str(body.get("model", ""))
        with deploy_lock:
            result = activate_model(config, db, name)
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @router.post("/api/models/backups/{backup_id}/revert")
    def api_backup_revert(backup_id: int):
        from .trainer import restore_backup

        with deploy_lock:
            result = restore_backup(config, db, backup_id)
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @router.get("/api/train/status")
    def train_status():
        return train_state

    @router.get("/api/train/mac-kit.zip")
    def download_mac_kit():
        from .macbundle import build_mac_kit

        path = build_mac_kit(config, db)
        return FileResponse(
            path,
            media_type="application/zip",
            filename="errata-mac-train.zip",
            background=BackgroundTask(os.unlink, path),
        )

    @router.post("/api/run-now")
    def run_now():
        scheduler.run_now()
        return JSONResponse({"ok": True, "detail": "scheduler run triggered"}, status_code=202)

    app.include_router(router)
    return app
