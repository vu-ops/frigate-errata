from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import requests

from .analyzer import Analyzer
from .db import Database
from .frigate_client import FrigateClient
from .harvester import Harvester

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, cfg: dict, db: Database):
        self.cfg = cfg
        self.db = db
        self.client = FrigateClient(cfg["frigate"])
        self.harvester = Harvester(cfg, db, self.client)
        self.analyzer = Analyzer(cfg, db)
        self.stop_event = threading.Event()
        self.run_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_stats: dict = {}
        self.last_run: float = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, name="errata-scheduler", daemon=True)
        self.thread.start()
        logger.info("scheduler started")

    def stop(self) -> None:
        self.stop_event.set()
        self.run_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("scheduler stopped")

    def run_now(self) -> None:
        self.run_event.set()

    def _loop(self) -> None:
        interval = float(self.cfg["harvest"]["interval_minutes"]) * 60
        while not self.stop_event.is_set():
            self.run_event.clear()
            try:
                self.run_once()
            except Exception:
                logger.exception("scheduler run failed")
            self.run_event.wait(interval)

    def run_once(self) -> dict:
        started = time.time()
        stats: dict = {"harvested": 0}
        try:
            stats["harvested"] = self.harvester.run()
        except Exception:
            logger.exception("harvest failed")
        try:
            stats.update(self.analyzer.run())
        except Exception:
            logger.exception("analysis failed")
        try:
            stats["retired_snapshots"] = self._retention()
        except Exception:
            logger.exception("retention cleanup failed")
        try:
            self._maybe_notify()
        except Exception:
            logger.exception("notification failed")
        self.last_stats = stats
        self.last_run = started
        logger.info("scheduler run complete: %s", stats)
        return stats

    def _retention(self) -> int:
        keep = int(self.cfg["review"].get("auto_keep_per_label", 150))
        if keep <= 0:
            return 0
        now = time.time()
        last = self.db.kv_get("last_prune_at")
        if last and (now - float(last)) < 86400:
            return 0
        self.db.kv_set("last_prune_at", str(now))
        candidates = self.db.prune_candidates(keep)
        by_bucket: dict = {}
        for row in candidates:
            path = row["snapshot_path"]
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    logger.exception("failed to delete snapshot %s", path)
            self.db.clear_snapshot(row["id"])
            by_bucket[(row["bucket"], row["bucket_label"])] = (
                by_bucket.get((row["bucket"], row["bucket_label"]), 0) + 1
            )
        if candidates:
            logger.info(
                "nightly prune removed %d snapshots beyond the %d-per-label cap (%d buckets affected)",
                len(candidates),
                keep,
                len(by_bucket),
            )
            for (bucket, label), n in sorted(by_bucket.items()):
                logger.info("prune bucket %s/%s: removed %d", bucket, label, n)
        return len(candidates)

    def _maybe_notify(self) -> None:
        notifications = self.cfg["notifications"]
        if not notifications.get("enabled") or not notifications.get("discord_webhook"):
            return
        threshold = int(self.cfg["training"]["trigger_threshold"])
        pending = self.db.corrections_count()
        last_notified = self.db.kv_get("last_notified_at")
        last_value = int(float(last_notified)) if last_notified else 0
        if pending < threshold or (time.time() - last_value) < 86400:
            return
        message = (
            f"Errata: {pending} corrections collected (threshold {threshold}). "
            "Run the trainer to build a new model."
        )
        requests.post(
            notifications["discord_webhook"],
            json={"content": message},
            timeout=15,
        )
        self.db.kv_set("last_notified_at", str(time.time()))
        logger.info("sent correction threshold notification")
