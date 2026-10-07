from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .db import Database
from .frigate_client import FrigateClient

logger = logging.getLogger(__name__)


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

    def run(self) -> int:
        Path(self.snapshot_dir).mkdir(parents=True, exist_ok=True)
        raw = self.db.kv_get("last_processed_at")
        since = float(raw) if raw else time.time() - self.lookback_hours * 3600
        added = 0
        scanned = 0
        newest = since
        time_to = None
        prev_oldest = None
        for page in range(self.max_pages):
            events = self.client.events(since=since, before=time_to, limit=self.max_events)
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
        attributes = data.get("attrs") or []
        confidence = data.get("score")
        if confidence is None:
            confidence = event.get("top_score")
        box = data.get("box")
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
                "attributes": json.dumps(attributes) if attributes else "",
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "box": json.dumps(box) if box else "",
                "snapshot_path": snapshot_path,
            }
        )
        return True
