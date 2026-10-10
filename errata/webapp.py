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

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request, UploadFile
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

from . import basemodels
from . import genai_help as genai
from .controls import (
    CONTROLS_DEFAULT_DISABLED_KEY,
    effective_controls,
    effective_for_label,
    reset_all_disabled,
    set_control,
)
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
    "confirmed",
    "ignored",
    "all",
]

FLAG_REASONS = ["mismatch", "candidate", "low_confidence", "noise", "oversized"]

REVIEWED_STATUSES = {"corrected", "false_positive", "confirmed", "ignored"}

VIEW_STATUSES = [
    ("pending", "Pending review"),
    ("corrected", "Corrected"),
    ("false_positive", "False positive"),
    ("confirmed", "Auto-confirmed"),
    ("ignored", "Ignored"),
    ("all", "All events"),
]

BRAND_STATUSES = [("pending", "Pending"), ("confirmed", "Confirmed"),
                  ("rejected", "Rejected"), ("all", "All")]

BULK_STATUSES = ["confirmed", "false_positive", "ignored"]
BULK_STATUS_ACTIONS = {
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
    attribute_labels = list(config["labels"].get("attributes", []))
    queue_limit = int(config["review"]["queue_limit"])
    page_size = int(config["review"].get("page_size", 100) or 100)
    preview_scale = float(config["review"].get("preview_scale", 4.0) or 4.0)
    preview_imgsz = int(config["review"].get("preview_imgsz", 640) or 640)
    imgsz = int(config["training"].get("imgsz", 320))

    basemodels.seed_builtin_models(db)

    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["localts"] = format_ts
    templates.env.filters["humandur"] = format_duration
    app_version = os.environ.get("ERRATA_VERSION", "").strip() or "dev"
    templates.env.globals["app_version"] = app_version

    from .trainer import IMGSZ_CHOICES, MODEL_CHOICES

    templates.env.globals["model_choices"] = MODEL_CHOICES
    templates.env.globals["default_model_type"] = str(config["training"].get("model_type", "yolov9s"))
    templates.env.globals["imgsz_choices"] = IMGSZ_CHOICES
    templates.env.globals["default_imgsz"] = imgsz

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
    deploy_lock = threading.Lock()

    def render(request: Request, template: str, ctx: dict, status_code: int = 200):
        ctx.update({"request": request, "base": base})
        return templates.TemplateResponse(request, template, ctx, status_code=status_code)

    # ---- review queue -------------------------------------------------------

    @router.get("/", response_class=HTMLResponse)
    def dashboard(
        request: Request,
        camera: str = "",
        label: str = "",
        reason: str = "",
        corrected: str = "",
        status: str = "pending",
        brand: str = "",
        brand_status: str = "pending",
        brand_camera: str = "",
        page: int = 1,
        msg: str = "",
    ):
        purged = db.purge_missing_snapshots()
        if purged and not msg:
            msg = f"Removed {purged} event(s) whose snapshot file no longer exists."
        page = max(1, int(page or 1))
        rows = db.queue(
            status=status or "pending",
            camera=camera or None,
            label=label or None,
            reason=reason or None,
            corrected=corrected or None,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        total = db.queue_total(
            status=status or "pending",
            camera=camera or None,
            label=label or None,
            reason=reason or None,
            corrected=corrected or None,
        )
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        brand_rows = db.brand_queue(
            status=brand_status or "pending",
            brand=brand or None,
            camera=brand_camera or None,
            limit=page_size,
        )
        genai_enabled = genai.help_enabled(config)
        genai_cards = {}
        genai_brand_cards = {}
        if genai_enabled:
            event_cache = {}
            for row in rows:
                card = _genai_card(row, "object", row["label"])
                if card:
                    genai_cards[row["id"]] = card
            for b in brand_rows:
                ev = event_cache.get(b["event_id"])
                if ev is None:
                    ev = db.get_event(b["event_id"])
                    event_cache[b["event_id"]] = ev
                if ev is None:
                    continue
                card = _genai_card(ev, "brand", b["brand"])
                if card:
                    genai_brand_cards[b["id"]] = card
        ctx = {
            "rows": rows,
            "brand_rows": brand_rows,
            "genai_cards": genai_cards,
            "genai_brand_cards": genai_brand_cards,
            "labels": labels,
            "attributes": attribute_labels,
            "cameras": db.distinct_cameras(),
            "event_labels": db.distinct_labels(),
            "corrected_labels": db.corrected_labels(),
            "reasons": FLAG_REASONS,
            "statuses": REVIEW_STATUSES,
            "view_statuses": VIEW_STATUSES,
            "brand_statuses": BRAND_STATUSES,
            "reviewed_statuses": REVIEWED_STATUSES,
            "bulk_statuses": BULK_STATUSES,
            "ignored_view": (status or "pending") == "ignored",
            "genai_help": {
                "enabled": genai.help_enabled(config),
                "models": genai.help_settings(config).get("models", []) or [],
                "default_model": genai.help_settings(config).get("default_model", ""),
            },
            "preview_scale": preview_scale,
            "preview_imgsz": preview_imgsz,
            "page": page,
            "pages": pages,
            "total": total,
            "page_size": page_size,
            "filters": {
                "camera": camera,
                "label": label,
                "reason": reason,
                "corrected": corrected,
                "status": status or "pending",
                "brand": brand,
                "brand_status": brand_status or "pending",
                "brand_camera": brand_camera,
            },
            "pending_count": db.queue_count("pending"),
            "brand_pending_count": db.brand_count("pending"),
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
            "brand_counts": db.brands_by_status(),
            "brands_by_brand": db.brands_by_brand(),
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

    # ---- synonyms -----------------------------------------------------------

    @router.get("/synonyms", response_class=HTMLResponse)
    def synonyms_page(request: Request, msg: str = ""):
        baseline = dict((config.get("analysis", {}) or {}).get("synonyms", {}) or {})
        merged = effective_synonyms(config, db)
        additions = {
            label: [w for w in merged.get(label, []) if w not in baseline.get(label, [])]
            for label in set(list(baseline.keys()) + list(merged.keys()))
        }
        ctx = {
            "baseline": baseline,
            "additions": additions,
            "labels": list(dict.fromkeys(labels + attribute_labels)),
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
            f"{base}/synonyms?msg={quote('Synonym additions saved. Applied on the next analyzer run.')}",
            status_code=303,
        )

    @router.post("/synonyms/reset")
    def synonyms_reset():
        db.kv_delete(SYNONYMS_OVERRIDE_KEY)
        return RedirectResponse(
            f"{base}/synonyms?msg={quote('Additions cleared; packaged baseline restored.')}",
            status_code=303,
        )

    # ---- object review ------------------------------------------------------

    def _parse_box_form(raw: str) -> str | None:
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
            params["msg"] = f"Bulk set label '{new_label}' on {len(rows)} event(s)"
            return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)
        action = BULK_STATUS_ACTIONS.get(status)
        if action is None:
            raise HTTPException(status_code=400, detail=f"cannot bulk-apply status '{status}'")
        rows = db.pending_matching(camera or None, label or None, reason or None)
        for row in rows:
            apply_review(row, action, row["label"] if action == "confirm" else "")
        params["msg"] = f"Bulk set '{status}' on {len(rows)} object event(s)"
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    @router.post("/bulk/selected")
    def bulk_selected(
        event_ids: list[str] = Form(...),
        action: str = Form(...),
        label: str = Form(""),
        return_to: str = Form(""),
    ):
        if action not in ("confirm", "relabel", "false_positive", "ignore", "reset"):
            raise HTTPException(status_code=400, detail=f"unknown action '{action}'")
        if action == "relabel" and not label.strip():
            raise HTTPException(status_code=400, detail="pick a label to apply")
        params = dict(parse_qsl(return_to, keep_blank_values=True))
        params.pop("msg", None)
        count = 0
        for event_id in event_ids:
            row = db.get_event(event_id)
            if row is None:
                continue
            if action == "confirm":
                apply_review(row, "confirm", row["label"])
            elif action == "relabel":
                apply_review(row, "confirm", label.strip())
            else:
                apply_review(row, action)
            count += 1
        verb = {"confirm": "confirmed as detected", "relabel": f"relabelled to {label}",
                "false_positive": "marked false positive",
                "ignore": "ignored", "reset": "restored"}[action]
        params["msg"] = f"{count} selected event(s) {verb}"
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    # ---- brand review -------------------------------------------------------

    @router.post("/brand/{brand_id}")
    def brand_review(
        brand_id: int,
        action: str = Form(...),
        brand: str = Form(""),
        return_to: str = Form(""),
    ):
        if action == "reset":
            db.set_brand_status(brand_id, "pending")
            msg = f"Brand item {brand_id} reset to pending"
        elif action in ("confirm", "confirmed"):
            db.set_brand_status(brand_id, "confirmed")
            msg = f"Brand item {brand_id} confirmed"
        elif action in ("reject", "rejected"):
            db.set_brand_status(brand_id, "rejected")
            msg = f"Brand item {brand_id} rejected"
        else:
            raise HTTPException(status_code=400, detail=f"unknown action '{action}'")
        params = dict(parse_qsl(return_to, keep_blank_values=True))
        params.pop("msg", None)
        params["msg"] = msg
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    @router.post("/brands/bulk")
    def brand_bulk(
        brand: str = Form(""),
        camera: str = Form(""),
        status: str = Form("pending"),
        action: str = Form(...),
        return_to: str = Form(""),
    ):
        if action not in ("confirm", "rejected", "reset"):
            raise HTTPException(status_code=400, detail="bad brand bulk action")
        target = {"confirm": "confirmed", "rejected": "rejected", "reset": "pending"}[action]
        rows = db.brand_queue(status=status or "pending", brand=brand or None,
                              camera=camera or None, limit=5000)
        for row in rows:
            db.set_brand_status(row["id"], target)
        params = dict(parse_qsl(return_to, keep_blank_values=True))
        params.pop("msg", None)
        params["msg"] = f"Bulk {action} on {len(rows)} brand item(s)"
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    @router.post("/brands/bulk/selected")
    def brand_bulk_selected(
        brand_ids: list[int] = Form(...),
        action: str = Form(...),
        return_to: str = Form(""),
    ):
        if action not in ("confirm", "rejected", "reset"):
            raise HTTPException(status_code=400, detail="bad brand bulk action")
        target = {"confirm": "confirmed", "rejected": "rejected", "reset": "pending"}[action]
        count = 0
        for brand_id in brand_ids:
            db.set_brand_status(int(brand_id), target)
            count += 1
        params = dict(parse_qsl(return_to, keep_blank_values=True))
        params.pop("msg", None)
        params["msg"] = f"{count} selected brand item(s) updated"
        return RedirectResponse(f"{base}/?{urlencode(params)}", status_code=303)

    # ---- per-label controls -------------------------------------------------

    @router.get("/controls", response_class=HTMLResponse)
    def controls_page(request: Request, msg: str = ""):
        from .trainer import verification_counts

        all_labels = list(dict.fromkeys(labels + attribute_labels))
        controls = effective_controls(config, db, all_labels)
        counts = verification_counts(config, db)
        for label in all_labels:
            counts.setdefault(label, {"human": 0, "machine": 0})
        ctx = {
            "controls": controls,
            "labels": labels,
            "attributes": attribute_labels,
            "counts": counts,
            "global_keep": int(config["review"].get("auto_keep_per_label", 150)),
            "default_disabled": bool(db.kv_get(CONTROLS_DEFAULT_DISABLED_KEY) == "1"),
            "msg": msg,
        }
        return render(request, "controls.html", ctx)

    @router.post("/controls")
    async def controls_save(request: Request):
        form = await request.form()
        all_labels = list(dict.fromkeys(labels + attribute_labels))
        for label in all_labels:
            if f"present_{label}" not in form:
                continue
            mode = str(form.get(f"collect_mode_{label}", "all"))
            raw_keep = form.get(f"keep_{label}")
            keep = None
            if raw_keep not in (None, ""):
                try:
                    keep = max(0, int(str(raw_keep)))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"invalid snapshot limit for '{label}'")
            set_control(
                config, db, label,
                collect_mode=mode,
                search=form.get(f"search_{label}") is not None,
                human_verifications=str(form.get(f"human_verifications_{label}", "collect")),
                machine_verifications=str(form.get(f"machine_verifications_{label}", "collect")),
                keep=keep,
            )
        return RedirectResponse(
            f"{base}/controls?msg={quote('Label controls saved.')}", status_code=303
        )

    # ---- base models --------------------------------------------------------

    base_state = {
        "running": False,
        "name": None,
        "imgsz": None,
        "phase": None,
        "detail": "no export run yet",
        "ok": None,
        "started_at": None,
        "finished_at": None,
        "result": None,
    }

    def _base_worker(name: str, imgsz: int | None) -> None:
        base_state.update({
            "running": True, "name": name, "imgsz": imgsz, "phase": "starting",
            "detail": f"preparing {name}", "ok": None, "started_at": time.time(),
            "finished_at": None, "result": None,
        })
        try:
            result = basemodels.export_yolo_variant(
                config, db, name, imgsz=imgsz, on_progress=base_state.update,
            )
        except Exception as exc:
            logger.exception("base model export failed")
            base_state.update({
                "running": False, "ok": False, "finished_at": time.time(),
                "phase": "failed", "detail": str(exc), "result": None,
            })
            return
        if result.get("ok"):
            base_state.update({
                "running": False, "ok": True, "finished_at": time.time(),
                "phase": "done", "detail": f"activated {result.get('name') or name}",
                "result": result,
            })
        else:
            base_state.update({
                "running": False, "ok": False, "finished_at": time.time(),
                "phase": "failed", "detail": result.get("error") or "export failed",
                "result": result,
            })

    @router.get("/base-models", response_class=HTMLResponse)
    def base_models_page(request: Request, msg: str = ""):
        ctx = {
            "models": basemodels.list_base_models(db),
            "base_state": base_state,
            "msg": msg,
        }
        return render(request, "base_models.html", ctx)

    @router.post("/base-models/import-plus")
    def base_models_import_plus(
        key: str = Form(...),
        model_id: str = Form(...),
    ):
        result = basemodels.import_frigate_plus(config, db, key.strip(), model_id.strip())
        msg = f"Imported {result.get('name')}" if result.get("ok") else f"Import failed: {result.get('error')}"
        return RedirectResponse(f"{base}/base-models?msg={quote(msg)}", status_code=303)

    @router.post("/base-models/upload")
    async def base_models_upload(file: UploadFile, name: str = Form("")):
        data = await file.read()
        if len(data) > 500 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="file too large (max 500 MB)")
        result = basemodels.store_upload(config, db, file.filename or "model", data, name or None)
        msg = f"Uploaded {result.get('name')}" if result.get("ok") else f"Upload failed: {result.get('error')}"
        return RedirectResponse(f"{base}/base-models?msg={quote(msg)}", status_code=303)

    @router.post("/base-models/{name}/delete")
    def base_models_delete(name: str):
        result = basemodels.delete_base_model(db, name)
        msg = f"Deleted {name}" if result.get("ok") else f"Delete failed: {result.get('error')}"
        return RedirectResponse(f"{base}/base-models?msg={quote(msg)}", status_code=303)

    @router.post("/base-models/{name}/activate")
    def base_models_activate(name: str):
        with deploy_lock:
            result = basemodels.activate_base_model(config, db, name)
        if result.get("ok"):
            msg = f"Activated {name} in Frigate ({result.get('layout')} @ {result.get('imgsz')}px)"
        else:
            msg = f"Activate failed: {result.get('error')}"
        return RedirectResponse(f"{base}/base-models?msg={quote(msg)}", status_code=303)

    @router.post("/base-models/{name}/export-activate")
    def base_models_export_activate(name: str, imgsz: int = Form(0)):
        from .trainer import ALL_MODELS, IMGSZ_CHOICES

        if name not in ALL_MODELS:
            return RedirectResponse(
                f"{base}/base-models?msg={quote(f'{name} is not a packaged YOLO variant')}",
                status_code=303,
            )
        if importlib.util.find_spec("ultralytics") is None:
            return RedirectResponse(
                f"{base}/base-models?msg={quote('ultralytics is not installed in this container')}",
                status_code=303,
            )
        if train_state["running"] or base_state["running"]:
            return RedirectResponse(
                f"{base}/base-models?msg={quote('another export or training run is already in progress')}",
                status_code=303,
            )
        chosen = imgsz if imgsz in IMGSZ_CHOICES else None
        threading.Thread(
            target=_base_worker, args=(name, chosen),
            name="errata-base-export", daemon=True,
        ).start()
        return RedirectResponse(
            f"{base}/base-models?msg={quote(f'Exporting {name} to ONNX and activating in Frigate…')}",
            status_code=303,
        )

    @router.get("/api/base-models/status")
    def base_models_status():
        return JSONResponse(base_state)

    # ---- snapshots / crop preview ------------------------------------------

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

    def _effective_box(row) -> tuple[str, str]:
        box = row["box"] if row else ""
        label = row["label"] if row else ""
        corr = db.latest_correction(row["id"]) if row else None
        if corr:
            if corr["correct_label"] == "false_positive":
                box = ""
            elif corr["box"]:
                box = corr["box"]
                label = corr["correct_label"] or label
        return box, label

    def _snapshot_or_404(event_id: str) -> str:
        row = db.get_event(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        path = row["snapshot_path"]
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=404, detail="snapshot not available")
        return path

    def _box_values(raw):
        if not raw:
            return None
        try:
            return tuple(float(v) for v in json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _entry_result(entry: dict) -> dict:
        return {
            "ok": True,
            "model": entry.get("model"),
            "matches": entry.get("matches"),
            "label": entry.get("label"),
            "description": entry.get("description"),
            "confidence": entry.get("confidence"),
            "box": entry.get("box") or {},
        }

    def _genai_payload(row, result: dict, cached: bool, created_at) -> dict:
        """Attach preview-crop geometry to a GenAI result for the overlay UI."""
        from .imaging import box_in_crop, preview_geometry

        detector = _box_values(row["box"])
        gbox = result.get("box") or {}
        genai_frame = None
        if gbox.get("found") and all(gbox.get(k) is not None for k in ("x", "y", "w", "h")):
            genai_frame = (float(gbox["x"]), float(gbox["y"]), float(gbox["w"]), float(gbox["h"]))
        payload = dict(result)
        payload["cached"] = cached
        payload["created_at"] = created_at
        payload["detector_frame"] = detector
        payload["genai_frame"] = genai_frame
        payload["detector_crop"] = None
        payload["genai_crop"] = None
        path = row["snapshot_path"]
        if path and Path(path).is_file():
            try:
                from PIL import Image

                with Image.open(path) as im:
                    fw, fh = im.size
                x0, y0, side = preview_geometry(detector, fw, fh, scale=preview_scale)
                payload["detector_crop"] = box_in_crop(detector, x0, y0, side, fw, fh)
                if genai_frame is not None:
                    payload["genai_crop"] = box_in_crop(genai_frame, x0, y0, side, fw, fh)
            except Exception:
                logger.exception("genai geometry failed for %s", row["id"])
        return payload

    def _genai_card(row, kind: str, expected: str):
        """Most-recent cached GenAI result for a card, as overlay geometry."""
        cache = genai.load_cache(row["genai"])
        found = genai.most_recent(cache, kind, expected)
        if not found:
            return None
        _key, entry = found
        payload = _genai_payload(row, _entry_result(entry), True, entry.get("created_at"))
        label = str(payload.get("label") or "").strip()
        payload["usable"] = bool(label) and label.lower() != "none"
        return payload

    @router.get("/api/snapshots/{event_id}.jpg")
    def snapshot(event_id: str):
        path = _snapshot_or_404(event_id)
        row = db.get_event(event_id)
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        box, label = _effective_box(row)
        if box:
            annotated = _annotate(path, box, label or "")
            if annotated is not None:
                return Response(content=annotated, media_type="image/jpeg", headers=no_cache)
        return FileResponse(path, media_type="image/jpeg", headers=no_cache)

    @router.get("/api/snapshots/{event_id}/clean.jpg")
    def snapshot_clean(event_id: str):
        path = _snapshot_or_404(event_id)
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        return FileResponse(path, media_type="image/jpeg", headers=no_cache)

    @router.get("/api/snapshots/{event_id}/crop.jpg")
    def snapshot_crop(event_id: str):
        """The actual training crop for this event, with the box drawn on it."""
        from .imaging import region_crop_pil

        path = _snapshot_or_404(event_id)
        row = db.get_event(event_id)
        box, label = _effective_box(row)
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        try:
            from PIL import Image, ImageDraw

            img = Image.open(path).convert("RGB")
            parsed = None
            if box:
                try:
                    parsed = tuple(float(v) for v in json.loads(box))
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed = None
            crop, box_c = region_crop_pil(img, parsed, imgsz)
            if box_c:
                w_px, h_px = crop.size
                x, y, w, h = box_c
                draw = ImageDraw.Draw(crop)
                draw.rectangle(
                    [int(x * w_px), int(y * h_px), int((x + w) * w_px), int((y + h) * h_px)],
                    outline=(0, 220, 255), width=2,
                )
                if label:
                    draw.rectangle([2, 2, 8 * len(label) + 8, 15], fill=(0, 0, 0))
                    draw.text((4, 3), label, fill=(0, 220, 255))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            return Response(content=buf.getvalue(), media_type="image/jpeg", headers=no_cache)
        except Exception:
            logger.exception("failed to build crop preview for %s", event_id)
            return FileResponse(path, media_type="image/jpeg", headers=no_cache)

    @router.get("/api/snapshots/{event_id}/preview.jpg")
    def snapshot_preview(event_id: str, raw: int = 0):
        """Wider-context preview around the object (training crop is tighter).

        ``raw=1`` returns the crop without any baked-in box so the client can
        draw its own Frigate/GenAI overlays.
        """
        from .imaging import region_crop_pil_fit as region_crop_pil

        path = _snapshot_or_404(event_id)
        row = db.get_event(event_id)
        box, label = _effective_box(row)
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        try:
            from PIL import Image, ImageDraw

            img = Image.open(path).convert("RGB")
            parsed = None
            if box:
                try:
                    parsed = tuple(float(v) for v in json.loads(box))
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed = None
            crop, box_c = region_crop_pil(img, parsed, preview_imgsz, scale=preview_scale)
            if box_c and not raw:
                w_px, h_px = crop.size
                x, y, w, h = box_c
                draw = ImageDraw.Draw(crop)
                draw.rectangle(
                    [int(x * w_px), int(y * h_px), int((x + w) * w_px), int((y + h) * h_px)],
                    outline=(0, 220, 255), width=2,
                )
                if label:
                    draw.rectangle([2, 2, 8 * len(label) + 8, 15], fill=(0, 0, 0))
                    draw.text((4, 3), label, fill=(0, 220, 255))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            return Response(content=buf.getvalue(), media_type="image/jpeg", headers=no_cache)
        except Exception:
            logger.exception("failed to build preview for %s", event_id)
            return FileResponse(path, media_type="image/jpeg", headers=no_cache)

    @router.post("/api/genai_help/{event_id}")
    async def genai_help_event(event_id: str, request: Request):
        if not genai.help_enabled(config):
            return JSONResponse(
                {"ok": False, "error": "GenAI Help is disabled (no OpenRouter API key configured)"},
                status_code=503,
            )
        body = await request.json()
        kind = str(body.get("kind") or "object")
        model = str(body.get("model") or "").strip()
        configured = genai.help_settings(config).get("models", []) or []
        allowed_ids = [m.get("id") for m in configured if isinstance(m, dict)]
        if model and model not in allowed_ids:
            return JSONResponse({"ok": False, "error": f"model '{model}' is not configured"}, status_code=400)
        row = db.get_event(event_id)
        if row is None:
            return JSONResponse({"ok": False, "error": "event not found"}, status_code=404)
        path = row["snapshot_path"]
        if not path or not Path(path).is_file():
            return JSONResponse({"ok": False, "error": "snapshot not available"}, status_code=404)
        if kind == "brand":
            expected = str(body.get("expected") or "").strip()
            allowed = attribute_labels
        else:
            expected = row["label"]
            allowed = labels
        use_model = model or genai.help_settings(config).get("default_model") or ""
        cache = genai.load_cache(row["genai"])
        key = genai.cache_key(kind, expected, use_model)
        cached_entry = cache.get(key)
        if isinstance(cached_entry, dict):
            payload = _genai_payload(row, _entry_result(cached_entry), True, cached_entry.get("created_at"))
            return JSONResponse(payload, status_code=200)
        result = genai.analyze(config, path, kind, expected, allowed, model or None, box=row["box"])
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        entry = {
            "created_at": time.time(),
            "model": result.get("model"),
            "matches": result.get("matches"),
            "label": result.get("label"),
            "description": result.get("description"),
            "confidence": result.get("confidence"),
            "box": result.get("box"),
        }
        cache[key] = entry
        db.set_event_genai(event_id, json.dumps(cache))
        return JSONResponse(_genai_payload(row, result, False, entry["created_at"]), status_code=200)

    @router.get("/api/queue.json")
    def queue_json(status: str = "pending"):
        rows = db.queue(status=status, limit=queue_limit)
        brands = db.brand_queue(status=status if status != "new" else "pending", limit=queue_limit)
        return JSONResponse(
            {
                "generated_at": time.time(),
                "status": status,
                "count": len(rows),
                "events": [dict(r) for r in rows],
                "brands": [dict(r) for r in brands],
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
            "brands_pending": db.brand_count("pending"),
            "corrections": db.corrections_count(),
        }

    # ---- training -----------------------------------------------------------

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
                "running": True, "started_at": time.time(), "finished_at": None,
                "ok": None, "detail": "training", "phase": "starting",
                "model_type": chosen, "imgsz": chosen_imgsz,
                "epoch": None, "epochs": None, "progress": None, "metrics": None,
            }
        )
        try:
            from .trainer import export_dataset, run_training

            train_state["detail"] = "exporting dataset"
            train_state["phase"] = "exporting dataset"
            exported = export_dataset(config, db, imgsz=imgsz)
            train_state["detail"] = "training"
            result = run_training(
                config, db, device=None, on_progress=train_state.update,
                model_type=model_type, imgsz=imgsz,
            )
            result["exported"] = exported
            db.kv_set("last_train_at", str(time.time()))
            train_state.update(
                {
                    "running": False, "finished_at": time.time(), "ok": True,
                    "detail": json.dumps(result), "phase": "done", "progress": 1.0,
                }
            )
        except Exception as exc:
            logger.exception("training job failed")
            train_state.update(
                {
                    "running": False, "finished_at": time.time(), "ok": False,
                    "detail": str(exc), "phase": "failed",
                }
            )

    @router.post("/api/train")
    def start_train(model_type: str = "", imgsz: int = 0):
        from .trainer import ALL_MODELS, IMGSZ_CHOICES, trained_class_map

        model_type = model_type.strip()
        if model_type and model_type not in ALL_MODELS:
            base = db.base_model_get(model_type)
            if base is None:
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
        if not trained_class_map(config, db):
            return JSONResponse(
                {"ok": False, "detail": "no labels have a verification source set to 'Collect and Train'; enable one on the Controls page"},
                status_code=400,
            )
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

    # ---- reset --------------------------------------------------------------

    @router.post("/api/reset")
    async def api_reset(request: Request):
        if train_state["running"]:
            return JSONResponse(
                {"ok": False, "error": "a training run is in progress; try again when it finishes"},
                status_code=409,
            )
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
        else:
            form = await request.form()
            body = dict(form)
        confirm = str(body.get("confirm", "")).strip()
        if confirm != "RESET":
            return JSONResponse({"ok": False, "error": "type RESET to confirm"}, status_code=400)

        from .trainer import reset_training_data_files, activate_model

        with deploy_lock:
            return_model = str(body.get("return_model", "")).strip() or None
            result = reset_training_data_files(config, db, keep_model=return_model)
            if return_model:
                result["activate"] = activate_model(config, db, return_model)
        return JSONResponse(result)

    app.include_router(router)
    return app
