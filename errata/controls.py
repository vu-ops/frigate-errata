from __future__ import annotations

import logging

from .config import control_defaults

logger = logging.getLogger(__name__)

CONTROLS_DEFAULT_DISABLED_KEY = "controls_default_disabled"

DISABLED_DEFAULTS = {
    "collect_mode": "off",
    "search": False,
    "human_verifications": "collect",
    "machine_verifications": "collect",
    "keep": None,
}

_BOOL_FIELDS = ("search",)
_VALID_COLLECT = ("all", "review_only", "off")
_VALID_HUMAN = ("collect", "train")
_VALID_MACHINE = ("collect", "train", "train_only")


def _normalize(settings: dict) -> dict:
    out = dict(settings)
    out["collect_mode"] = out.get("collect_mode") if out.get("collect_mode") in _VALID_COLLECT else "all"
    for field in _BOOL_FIELDS:
        out[field] = bool(out.get(field))
    out["human_verifications"] = (
        out.get("human_verifications") if out.get("human_verifications") in _VALID_HUMAN else "train"
    )
    out["machine_verifications"] = (
        out.get("machine_verifications") if out.get("machine_verifications") in _VALID_MACHINE else "train"
    )
    keep = out.get("keep")
    try:
        out["keep"] = int(keep) if keep not in (None, "") else None
    except (TypeError, ValueError):
        out["keep"] = None
    return out


def controls_default_disabled(db) -> bool:
    return db.kv_get(CONTROLS_DEFAULT_DISABLED_KEY) == "1"


_PASSTHROUGH = ("collect_mode", "keep", "human_verifications", "machine_verifications")


def effective_controls(config: dict, db, labels: list[str]) -> dict[str, dict]:
    """Resolve per-label settings: config defaults, reset-disabled defaults, DB overrides.

    Precedence: built-in/config default  <  reset-disabled default (if set)  <  DB row.
    """
    base = control_defaults(config)
    if controls_default_disabled(db):
        base = {**base, **DISABLED_DEFAULTS}
    stored = db.all_label_settings()
    result: dict[str, dict] = {}
    for label in labels:
        row = stored.get(label) or {}
        merged = {**base}
        for key, value in row.items():
            if key in _PASSTHROUGH or key in _BOOL_FIELDS:
                merged[key] = value
        result[label] = _normalize(merged)
    return result


def effective_for_label(config: dict, db, label: str) -> dict:
    return effective_controls(config, db, [label]).get(label, _normalize(control_defaults(config)))


def set_control(config: dict, db, label: str, **fields) -> dict:
    """Update one or more control fields, preserving the rest of the effective set."""
    current = effective_for_label(config, db, label)
    current.update({k: v for k, v in fields.items() if k in _PASSTHROUGH or k in _BOOL_FIELDS})
    db.set_label_settings(
        label,
        collect_mode=current["collect_mode"],
        search=current["search"],
        human_verifications=current["human_verifications"],
        machine_verifications=current["machine_verifications"],
        keep=current.get("keep"),
    )
    return current


def reset_all_disabled(config: dict, db, labels: list[str]) -> None:
    """Write disabled rows for every label (used by the reset flow)."""
    for label in labels:
        db.set_label_settings(
            label,
            collect_mode="off",
            search=False,
            human_verifications="collect",
            machine_verifications="collect",
            keep=None,
        )
    db.kv_set(CONTROLS_DEFAULT_DISABLED_KEY, "1")
