from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .db import Database
from .frigate_client import FrigateClient

logger = logging.getLogger(__name__)


def _flatten_strings(value) -> list[str]:
    """Collect all string leaves from a nested JSON value."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_strings(item))
    return out


def extract_brands(event: dict, attribute_labels: list[str]) -> list[str]:
    """Find configured brand attributes present on a Frigate event.

    Frigate stores classification attributes in the event's ``data`` JSON
    (either under ``attributes``/``current_attributes`` or keyed by model name)
    and can also set a top-level ``sub_label``. We match any string leaf against
    the configured attribute list, case-insensitively.
    """
    wanted = {a.lower(): a for a in attribute_labels}
    found: dict[str, None] = {}

    sub_label = event.get("sub_label")
    if isinstance(sub_label, (list, tuple)) and sub_label:
        sub_label = sub_label[0]
    if isinstance(sub_label, str) and sub_label.strip().lower() in wanted:
        found[wanted[sub_label.strip().lower()]] = None

    data = event.get("data") or {}
    for text in _flatten_strings(data):
        key = text.strip().lower()
        if key in wanted:
            found[wanted[key]] = None
    return sorted(found)


class Harvester:
    def __init__(self, cfg: dict, db: Database, client: FrigateClient):
        self.cfg = cfg
        self.db = db
        self.client = client
        self.snapshot_dir = cfg["frigate"]["snapshot_dir"]
        harvest = cfg["harvest"]
        self.lookback_hours = float(harvest["lookback_hours"])
        self.max_events = int(harvest["max_events_per_run"])
        self.max_pages = max(1, int(harvest.get("max_pages", 20)))
        self.attribute_labels = list((cfg.get("labels", {}) or {}).get("attributes", []) or [])
        # Labels to request from Frigate: object classes not turned off.
        self.collect_labels = list(
            (cfg.get("labels", {}) or {}).get("track", []) or []
        )

    def run(self) -> int:
        Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        # Imported lazily to avoid a config <-> controls import cycle at module load.
        from .controls import effective_controls

        controls = effective_controls(self.cfg, self.db, self.collect_labels)
        if not self.collect_labels:
            logger.info("no labels configured to track; skipping harvest")
            return 0
        # Monitor Only (collect_mode "off") labels are still harvested; the
        # analyzer routes their events straight to ignored. Only brand review is
        # gated below (no brand items for monitor-only labels).
        self._collected = {
            label for label in self.collect_labels
            if controls.get(label, {}).get("collect_mode", "all") != "off"
        }
        labels_param = self.collect_labels

        raw = self.db.kv_get("last_processed_at")
        since = float(raw) if raw else time.time() - self.lookback_hours * 3600
        added = 0
        scanned = 0
        newest = since
        time_to = None
        prev_oldest = None
        for page in range(self.max_pages):
            events = self.client.events(
                since=since, before=time_to, limit=self.max_events, labels=labels_param
            )
            completed = [e for e in events if e.get("end_time")]
            for event in events:
                try:
                    if self._process_event(event):
                        added += 1
                    start = event.get("start_time") or 0
                    if event.get("end_time") and start > newest:
                        newest = start
                except Exception:
                    logger.exception("failed to process event %s", event.get("id"))
            scanned += len(events)
            if len(completed) < self.max_events:
                break
            oldest = min(e["start_time"] for e in completed)
            if prev_oldest is not None and oldest >= prev_oldest:
                logger.warning("harvest paging made no progress, stopping at page %d", page + 1)
                break
            prev_oldest = oldest
            logger.info("harvest page %d done, paging deeper (oldest %s)", page + 1, oldest)
            time_to = oldest
        if newest > since:
            self.db.kv_set("last_processed_at", str(newest))
        logger.info("harvested %d new events (scanned %d)", added, scanned)
        return added

    def _process_event(self, event: dict) -> bool:
        event_id = event.get("id")
        if not event_id or not event.get("end_time") or self.db.has_event(event_id):
            return False
        data = event.get("data") or {}
        description = data.get("description") or ""
        confidence = data.get("score")
        if confidence is None:
            confidence = event.get("top_score")
        box = data.get("box")
        sub_label = event.get("sub_label")
        if isinstance(sub_label, (list, tuple)) and sub_label:
            sub_label = sub_label[0]
        sub_label = sub_label if isinstance(sub_label, str) else ""
        collected = getattr(self, "_collected", None)
        brands = (
            extract_brands(event, self.attribute_labels)
            if collected is None or event.get("label") in collected
            else []
        )
        snapshot_path = ""
        if event.get("has_snapshot"):
            dest = str(Path(self.snapshot_dir) / f"{event_id}.jpg")
            if self.client.download_snapshot(event_id, dest):
                snapshot_path = dest
        self.db.upsert_event(
            {
                "id": event_id,
                "camera": event.get("camera", "unknown"),
                "label": event.get("label", "unknown"),
                "confidence": confidence,
                "description": description,
                "attributes": json.dumps(data.get("attributes") or brands or {}),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "box": json.dumps(box) if box else "",
                "snapshot_path": snapshot_path,
                "sub_label": sub_label,
            }
        )
        for brand in brands:
            self.db.insert_brand(event_id, brand, source="frigate")
        return True
