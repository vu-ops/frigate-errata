from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def help_settings(cfg: dict) -> dict:
    return (cfg.get("review", {}) or {}).get("genai_help", {}) or {}


def help_enabled(cfg: dict) -> bool:
    gh = help_settings(cfg)
    return bool(gh.get("enabled", True)) and bool(str(gh.get("api_key") or "").strip())


def _image_data_url(image_path: str, max_px: int) -> str:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_px) / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _schema() -> dict:
    return {
        "name": "event_help",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "best allowed label, or 'none'"},
                "description": {"type": "string", "description": "one-sentence description"},
                "confidence": {"type": "number", "description": "0..1 confidence"},
                "box": {
                    "type": "object",
                    "properties": {
                        "found": {"type": "boolean"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "w": {"type": "number"},
                        "h": {"type": "number"},
                    },
                    "required": ["found", "x", "y", "w", "h"],
                    "additionalProperties": False,
                },
            },
            "required": ["label", "description", "confidence", "box"],
            "additionalProperties": False,
        },
    }


def _prompt(kind: str, expected: str, allowed: list[str]) -> str:
    labels = ", ".join(allowed)
    if kind == "brand":
        return (
            f"This is a security-camera snapshot. A detector reports the object as "
            f"'{expected}'. Decide whether the object carries the delivery brand "
            f"'{expected}'. Choose the single best matching label from: {labels} "
            f"(use \"none\" if the brand is not present). Give a one-sentence description. "
            f"For a brand, bounding boxes are not needed: set box.found=false."
        )
    return (
        f"This is a security-camera snapshot. A detector labeled the object as "
        f"'{expected}'. Identify the single main object of interest. Choose the best "
        f"label from: {labels} (use \"none\" if none apply). Give a one-sentence "
        f"description. If there is exactly one clear object, return its bounding box as "
        f"normalized x,y,w,h with a top-left origin and values 0..1 of the full frame; "
        f"otherwise set box.found=false."
    )


def analyze(cfg: dict, image_path: str, kind: str, expected: str,
            allowed_labels: list[str], model: str | None = None) -> dict:
    """Ask an OpenRouter vision model for a second opinion on one snapshot."""
    gh = help_settings(cfg)
    key = str(gh.get("api_key") or "").strip()
    if not key:
        return {"ok": False, "error": "GenAI Help is not configured (no OpenRouter API key)"}
    if not Path(image_path).is_file():
        return {"ok": False, "error": "snapshot not available"}
    base = str(gh.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
    timeout = float(gh.get("timeout") or 60)
    max_px = int(gh.get("max_image_px") or 1024)
    use_model = model or gh.get("default_model")
    if not use_model:
        return {"ok": False, "error": "no model configured"}

    try:
        data_url = _image_data_url(image_path, max_px)
    except Exception as exc:
        logger.exception("genai help: could not read snapshot")
        return {"ok": False, "error": f"could not read snapshot: {exc}"}

    payload = {
        "model": use_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(kind, expected, allowed_labels)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {"type": "json_schema", "json_schema": _schema()},
        "provider": {"require_parameters": True},
        "temperature": 0,
    }
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {"ok": False, "error": f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}"}
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
    except Exception as exc:
        logger.exception("genai help request failed")
        return {"ok": False, "error": str(exc)}

    box = data.get("box") or {}
    return {
        "ok": True,
        "model": use_model,
        "label": str(data.get("label") or ""),
        "description": str(data.get("description") or ""),
        "confidence": data.get("confidence"),
        "box": {k: box.get(k) for k in ("found", "x", "y", "w", "h")},
    }