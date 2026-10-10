from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# The packaged baseline lives next to this module and ships in the image.
SETTINGS_PATH = Path(__file__).with_name("settings.yaml")

# Keys whose list values are appended (baseline first, then user additions,
# deduped) instead of replaced. Currently only per-label synonym lists.
APPEND_LIST_PARENTS = ("synonyms",)


def expand_env(text: str) -> str:
    def repl(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return ENV_PATTERN.sub(repl, text)


def load_baseline() -> dict:
    """Read the packaged settings.yaml that ships inside the container."""
    try:
        text = SETTINGS_PATH.read_text(encoding="utf-8")
    except OSError:
        logger.warning("packaged settings.yaml not found at %s; using empty baseline", SETTINGS_PATH)
        return {}
    return yaml.safe_load(text) or {}


def _merge_append_into(parent: dict, key: str, base: dict, override: dict) -> None:
    """Append-only merge for a dict of label -> list[str] (e.g. synonyms)."""
    base_map = base.get(key) or {}
    override_map = override.get(key) or {}
    if not isinstance(base_map, dict) or not isinstance(override_map, dict):
        parent[key] = override_map or base_map
        return
    merged: dict = {k: list(v) if isinstance(v, list) else v for k, v in base_map.items()}
    for label, values in override_map.items():
        if not isinstance(values, list):
            merged[label] = values
            continue
        existing = merged.get(label)
        if not isinstance(existing, list):
            merged[label] = list(values)
            continue
        seen = {str(v).strip().lower() for v in existing}
        for value in values:
            norm = str(value).strip().lower()
            if norm and norm not in seen:
                existing.append(value)
                seen.add(norm)
    parent[key] = merged


def deep_merge(base: dict, override: dict) -> dict:
    """Merge user overrides over the baseline.

    Special-cases synonym-style maps so a user list is *appended* to the
    packaged baseline rather than replacing it.
    """
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            if key in APPEND_LIST_PARENTS:
                _merge_append_into(merged, key, merged, value)
            else:
                merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def find_config_path(explicit: str | None = None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("ERRATA_CONFIG"),
        "/config/errata.yaml",
        str(Path.cwd() / "config.yaml"),
    ]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def load_config(explicit: str | None = None) -> dict:
    """Load config: packaged settings.yaml first, then user overrides."""
    config = load_baseline()
    path = find_config_path(explicit)
    if path is None:
        logger.warning("no user config file found, using packaged settings.yaml only")
        return config
    text = expand_env(Path(path).read_text(encoding="utf-8"))
    data = yaml.safe_load(text) or {}
    config = deep_merge(config, data)
    logger.info("loaded config from %s (over packaged settings.yaml)", path)
    return config


def tracked_labels(config: dict) -> list[str]:
    return list((config.get("labels", {}) or {}).get("track", []) or [])


def attribute_labels(config: dict) -> list[str]:
    return list((config.get("labels", {}) or {}).get("attributes", []) or [])


def control_defaults(config: dict) -> dict:
    controls = (config.get("labels", {}) or {}).get("controls", {}) or {}
    return {
        "collect_mode": "all",
        "search": False,
        "include_training": True,
        "auto_confirm": True,
        "pseudo_labels": True,
        "keep": None,
        **(controls.get("defaults") or {}),
    }


def reset_controls_disabled(config: dict) -> bool:
    controls = (config.get("labels", {}) or {}).get("controls", {}) or {}
    return bool(controls.get("reset_disabled", True))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )
