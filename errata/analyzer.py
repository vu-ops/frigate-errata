from __future__ import annotations

import json
import logging
import re
import time

from .db import Database
from .geometry import is_oversized_box, is_plausible_box
from .vocab import label_in_text

logger = logging.getLogger(__name__)

PRIORITY_MISMATCH_HIGH = 4
PRIORITY_MISMATCH = 3
PRIORITY_NOISE = 2
PRIORITY_OVERSIZED = 2
PRIORITY_LOW_CONFIDENCE = 1


class Analyzer:
    def __init__(self, cfg: dict, db: Database):
        self.cfg = cfg
        self.db = db
        analysis = cfg["analysis"]
        self.vocabulary_coherence = bool(analysis["vocabulary_coherence"])
        self.low_confidence_threshold = float(analysis["low_confidence_threshold"])
        self.confirm_on_coherent = bool(analysis.get("confirm_on_coherent_description", True))
        self.auto_skip_implausible = bool(
            analysis.get("auto_skip_implausible_boxes", True)
        )
        oversized_cfg = analysis.get("oversized_box", {}) or {}
        self.oversized_enabled = bool(oversized_cfg.get("enabled", True))
        self.oversized_max_area = float(oversized_cfg.get("max_area", 0.4))
        self.safe_labels = set(analysis["safe_labels"])
        self.file_synonyms = dict(analysis.get("synonyms", {}) or {})
        self.synonyms = dict(self.file_synonyms)
        self.track_labels = list(cfg["labels"]["track"])
        groups = analysis.get("suggestion_groups", {}) or {}
        self.suggestion_groups = dict(groups)
        self.noise_count = int(analysis["noise"]["count"])
        self.noise_window_hours = float(analysis["noise"]["window_hours"])
        harvest = cfg["harvest"]
        self.min_confidence = float(harvest["min_confidence"])
        self.max_confidence = float(harvest["max_confidence"])

    def _refresh_synonyms(self) -> None:
        raw = self.db.kv_get("synonyms_override")
        if raw is not None:
            try:
                self.synonyms = json.loads(raw) or {}
                return
            except json.JSONDecodeError:
                logger.warning("invalid synonyms_override in database, using file synonyms")
        self.synonyms = dict(self.file_synonyms)

    def run(self, statuses: tuple[str, ...] = ("new",)) -> dict:
        self._refresh_synonyms()
        purged = self.db.purge_missing_snapshots()
        if purged:
            logger.info("purged %d event(s) whose snapshot file is missing", purged)
        rows = self.db.pending_for_analysis(statuses)
        stats = {
            "scanned": len(rows),
            "mismatch": 0,
            "low_confidence": 0,
            "noise": 0,
            "oversized": 0,
            "ignored": 0,
            "confirmed": 0,
            "skipped": 0,
        }
        now = time.time()
        for row in rows:
            try:
                outcome = self._evaluate(row, now)
            except Exception:
                logger.exception("failed to analyze event %s", row["id"])
                continue
            if outcome is None:
                continue
            stats[outcome] = stats.get(outcome, 0) + 1
        logger.info("analysis complete: %s", stats)
        return stats

    def _suggest_labels(self, label: str, description: str) -> str:
        if not description:
            return ""
        matches = [
            candidate
            for candidate in self.track_labels
            if candidate != label and label_in_text(candidate, description, self.synonyms)
        ]
        label_group = self._group_of(label)
        if label_group:
            same = [m for m in matches if self._group_of(m) == label_group]
            other = [m for m in matches if self._group_of(m) != label_group]
            matches = same + other
        return ",".join(matches)

    def _group_of(self, label: str) -> str | None:
        for group, members in self.suggestion_groups.items():
            if label in members:
                return group
        return None

    def _evaluate(self, row, now: float) -> str | None:
        label = row["label"]
        confidence = row["confidence"]
        description = row["description"] or ""
        suggestion = ""

        if confidence is not None and confidence < self.min_confidence:
            self.db.set_status(row["id"], "ignored")
            return "ignored"

        # Degenerate detector boxes (e.g. a full-width sliver anchored on the
        # frame edge) are artifacts, not real objects. Keep them out of the
        # review queue and the training set entirely.
        if row["box"] and not is_plausible_box(row["box"]):
            if self.auto_skip_implausible:
                self.db.set_status(row["id"], "skipped", reviewed=True)
                return "skipped"
            self.db.set_flag(row["id"], "noise", PRIORITY_NOISE, "")
            return "noise"

        reason = None
        priority = 0
        description_confirms = False

        if self.vocabulary_coherence and description:
            description_confirms = label_in_text(label, description, self.synonyms)
            if not description_confirms:
                reason = "mismatch"
                if label in self.safe_labels:
                    priority = PRIORITY_NOISE
                elif confidence is not None and confidence >= 0.9:
                    priority = PRIORITY_MISMATCH_HIGH
                else:
                    priority = PRIORITY_MISMATCH
                suggestion = self._suggest_labels(label, description)

        # Oversized boxes (e.g. one box spanning two parked cars, common in IR
        # night frames) are geometrically legal but usually merged detections.
        # Never auto-confirm them; surface them for human review instead.
        if reason is None and self.oversized_enabled:
            if is_oversized_box(row["box"], self.oversized_max_area):
                reason = "oversized"
                priority = PRIORITY_OVERSIZED

        if reason is None and confidence is not None and confidence < self.low_confidence_threshold:
            if not (self.confirm_on_coherent and description_confirms):
                reason = "low_confidence"
                priority = PRIORITY_LOW_CONFIDENCE

        if reason is None and self.noise_count > 0:
            if not (self.confirm_on_coherent and description_confirms):
                window_start = now - self.noise_window_hours * 3600
                recent = self.db.count_recent(row["camera"], label, window_start)
                if recent >= self.noise_count:
                    reason = "noise"
                    priority = PRIORITY_NOISE

        if reason is None:
            self.db.set_status(row["id"], "confirmed")
            return "confirmed"

        self.db.set_flag(row["id"], reason, priority, suggestion)
        return reason
