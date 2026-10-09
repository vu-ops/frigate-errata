from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL,
    description TEXT,
    attributes TEXT,
    start_time REAL,
    end_time REAL,
    box TEXT,
    snapshot_path TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    flag_reason TEXT,
    priority INTEGER DEFAULT 0,
    suggested_label TEXT NOT NULL DEFAULT '',
    sub_label TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    reviewed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_camera_label ON events(camera, label);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT,
    image_path TEXT NOT NULL,
    correct_label TEXT NOT NULL,
    original_label TEXT,
    confidence REAL,
    camera TEXT,
    box TEXT,
    collected_at REAL NOT NULL,
    exported INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_corrections_exported ON corrections(exported);
CREATE INDEX IF NOT EXISTS idx_corrections_label ON corrections(correct_label);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path TEXT NOT NULL,
    created_at REAL NOT NULL,
    training_count INTEGER,
    metrics TEXT
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS frigate_backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    reason TEXT NOT NULL,
    model_path TEXT NOT NULL,
    file_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_brands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    brand TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'frigate',
    model TEXT,
    score REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewed_at REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_brands_status ON event_brands(status);
CREATE INDEX IF NOT EXISTS idx_event_brands_event ON event_brands(event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_event_brands_unique ON event_brands(event_id, brand);

CREATE TABLE IF NOT EXISTS label_settings (
    label TEXT PRIMARY KEY,
    collect_mode TEXT NOT NULL DEFAULT 'all',
    search INTEGER NOT NULL DEFAULT 0,
    include_training INTEGER NOT NULL DEFAULT 1,
    auto_confirm INTEGER NOT NULL DEFAULT 1,
    pseudo_labels INTEGER NOT NULL DEFAULT 1,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS base_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT,
    imgsz INTEGER,
    labels TEXT,
    trainable INTEGER NOT NULL DEFAULT 1,
    meta TEXT,
    created_at REAL NOT NULL
);
"""

class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
        if "suggested_label" not in cols:
            conn.execute(
                "ALTER TABLE events ADD COLUMN suggested_label TEXT NOT NULL DEFAULT ''"
            )
        if "sub_label" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN sub_label TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def has_event(self, event_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return row is not None

    def upsert_event(self, event: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO events
                    (id, camera, label, confidence, description, attributes,
                     start_time, end_time, box, snapshot_path, status, sub_label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
                """,
                (
                    event["id"],
                    event["camera"],
                    event["label"],
                    event.get("confidence"),
                    event.get("description"),
                    event.get("attributes"),
                    event.get("start_time"),
                    event.get("end_time"),
                    event.get("box"),
                    event.get("snapshot_path"),
                    event.get("sub_label") or "",
                    time.time(),
                ),
            )

    def get_event(self, event_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()

    def set_status(self, event_id: str, status: str, reviewed: bool = False) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE events SET status = ?, reviewed_at = ? WHERE id = ?",
                (status, time.time() if reviewed else None, event_id),
            )

    def set_flag(self, event_id: str, reason: str, priority: int, suggested_label: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE events
                SET status = 'flagged', flag_reason = ?, priority = ?, suggested_label = ?
                WHERE id = ?
                """,
                (reason, priority, suggested_label, event_id),
            )

    def pending_for_analysis(self, statuses: tuple[str, ...] = ("new",)) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as conn:
            return conn.execute(
                f"SELECT * FROM events WHERE status IN ({placeholders}) ORDER BY start_time DESC",
                tuple(statuses),
            ).fetchall()

    def queue(
        self,
        status: str = "pending",
        camera: str | None = None,
        label: str | None = None,
        reason: str | None = None,
        corrected: str | None = None,
        limit: int = 200,
    ) -> list[sqlite3.Row]:
        where = []
        params: list = []
        if status == "pending":
            where.append("status IN ('new', 'flagged')")
        elif status and status != "all":
            where.append("status = ?")
            params.append(status)
        if camera:
            where.append("camera = ?")
            params.append(camera)
        if label:
            where.append("label = ?")
            params.append(label)
        if reason:
            where.append("flag_reason = ?")
            params.append(reason)
        if corrected:
            where.append(
                "(SELECT c2.correct_label FROM corrections c2 WHERE c2.event_id = e.id "
                "ORDER BY c2.collected_at DESC, c2.id DESC LIMIT 1) = ?"
            )
            params.append(corrected)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            "SELECT e.*, "
            "(SELECT c.correct_label FROM corrections c WHERE c.event_id = e.id "
            "ORDER BY c.collected_at DESC, c.id DESC LIMIT 1) AS corrected_label "
            "FROM events e"
            + clause
            + " ORDER BY e.priority DESC, e.start_time DESC LIMIT ?"
        )
        params.append(limit)
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def pending_matching(
        self,
        camera: str | None = None,
        label: str | None = None,
        reason: str | None = None,
    ) -> list[sqlite3.Row]:
        where = ["status IN ('new', 'flagged')"]
        params: list = []
        if camera:
            where.append("camera = ?")
            params.append(camera)
        if label:
            where.append("label = ?")
            params.append(label)
        if reason:
            where.append("flag_reason = ?")
            params.append(reason)
        clause = " WHERE " + " AND ".join(where)
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM events" + clause + " ORDER BY priority DESC, start_time DESC",
                params,
            ).fetchall()

    def queue_count(self, status: str = "pending") -> int:
        if status == "pending":
            clause = "status IN ('new', 'flagged')"
            params: tuple = ()
        elif status == "all":
            clause = "1=1"
            params = ()
        else:
            clause = "status = ?"
            params = (status,)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM events WHERE {clause}", params
            ).fetchone()
        return row["n"]

    def counts_by_status(self) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM events GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def distinct_cameras(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT camera FROM events ORDER BY camera"
            ).fetchall()
        return [row["camera"] for row in rows]

    def distinct_labels(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT label FROM events ORDER BY label"
            ).fetchall()
        return [row["label"] for row in rows]

    def corrected_labels(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT correct_label FROM corrections ORDER BY correct_label"
            ).fetchall()
        return [row["correct_label"] for row in rows]

    def count_recent(self, camera: str, label: str, since: float) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE camera = ? AND label = ? AND start_time >= ?",
                (camera, label, since),
            ).fetchone()
        return row["n"]

    def corrections_by_group(self, column: str, since: float | None = None) -> list[sqlite3.Row]:
        if column not in ("camera", "correct_label"):
            raise ValueError(column)
        with self.connect() as conn:
            if since is None:
                return conn.execute(
                    f"SELECT {column} AS k, COUNT(*) AS n FROM corrections GROUP BY {column} ORDER BY n DESC"
                ).fetchall()
            return conn.execute(
                f"SELECT {column} AS k, COUNT(*) AS n FROM corrections WHERE collected_at >= ? GROUP BY {column} ORDER BY n DESC",
                (since,),
            ).fetchall()

    def corrections_count(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM corrections").fetchone()
        return row["n"]

    def corrections_unexported(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM corrections WHERE exported = 0 ORDER BY collected_at ASC"
            ).fetchall()

    def corrections_unexported_count(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM corrections WHERE exported = 0"
            ).fetchone()
        return row["n"]

    def corrections_mark_exported(self, ids: list[int]) -> None:
        if not ids:
            return
        with self.connect() as conn:
            conn.executemany(
                "UPDATE corrections SET exported = 1 WHERE id = ?",
                [(i,) for i in ids],
            )

    def corrections_reset_exported(self) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE corrections SET exported = 0")

    def insert_correction(self, correction: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO corrections
                    (event_id, image_path, correct_label, original_label,
                     confidence, camera, box, collected_at, exported)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    correction["event_id"],
                    correction["image_path"],
                    correction["correct_label"],
                    correction.get("original_label"),
                    correction.get("confidence"),
                    correction.get("camera"),
                    correction.get("box"),
                    time.time(),
                ),
            )

    def delete_corrections_for_event(self, event_id: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM corrections WHERE event_id = ?", (event_id,)
            )
            return cur.rowcount

    def latest_correction(self, event_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT correct_label, box FROM corrections WHERE event_id = ? "
                "ORDER BY collected_at DESC, id DESC LIMIT 1",
                (event_id,),
            ).fetchone()

    def reset_event(self, event_id: str) -> None:
        """Undo a review: drop any correction and return the event to the queue."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT flag_reason FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            status = "flagged" if row and row["flag_reason"] else "new"
            conn.execute("DELETE FROM corrections WHERE event_id = ?", (event_id,))
            conn.execute(
                "UPDATE events SET status = ?, reviewed_at = NULL WHERE id = ?",
                (status, event_id),
            )

    def clear_snapshot(self, event_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE events SET snapshot_path = '' WHERE id = ?", (event_id,)
            )

    def delete_event(self, event_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM corrections WHERE event_id = ?", (event_id,))
            conn.execute("DELETE FROM event_brands WHERE event_id = ?", (event_id,))
            conn.execute("DELETE FROM events WHERE id = ?", (event_id,))

    def purge_missing_snapshots(self) -> int:
        """Delete events (and their corrections) whose snapshot file is gone.

        Covers both cleared paths (nightly retention sets snapshot_path = '')
        and stale paths whose file no longer exists, so the queue and dataset
        never reference images that cannot be shown or trained on. Cascades to
        corrections and brand review rows.
        """
        with self.connect() as conn:
            rows = conn.execute("SELECT id, snapshot_path FROM events").fetchall()
            ids = [
                r["id"]
                for r in rows
                if not r["snapshot_path"] or not Path(r["snapshot_path"]).is_file()
            ]
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM corrections WHERE event_id IN ({marks})", chunk)
                conn.execute(f"DELETE FROM event_brands WHERE event_id IN ({marks})", chunk)
                conn.execute(f"DELETE FROM events WHERE id IN ({marks})", chunk)
        return len(ids)

    def set_snapshot(self, event_id: str, path: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE events SET snapshot_path = ? WHERE id = ?", (path, event_id)
            )

    def model_version_insert(self, model_path: str, training_count: int, metrics: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO model_versions (model_path, created_at, training_count, metrics) VALUES (?, ?, ?, ?)",
                (model_path, time.time(), training_count, metrics),
            )

    def model_versions(self, limit: int = 10) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def frigate_backup_insert(self, created_at: float, reason: str, model_path: str, file_path: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO frigate_backups (created_at, reason, model_path, file_path) VALUES (?, ?, ?, ?)",
                (created_at, reason, model_path, file_path),
            )
            return int(cur.lastrowid)

    def frigate_backups(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM frigate_backups ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def frigate_backup_get(self, backup_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM frigate_backups WHERE id = ?",
                (backup_id,),
            ).fetchone()

    def frigate_backups_prune(self, keep: int) -> list[str]:
        """Delete config-backup rows beyond the newest `keep`; return their files.

        A keep of 0 or less disables pruning and leaves every backup in place.
        """
        if keep <= 0:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, file_path FROM frigate_backups ORDER BY created_at DESC"
            ).fetchall()
            stale = rows[keep:]
            if not stale:
                return []
            conn.executemany(
                "DELETE FROM frigate_backups WHERE id = ?",
                [(row["id"],) for row in stale],
            )
        return [row["file_path"] for row in stale]

    def prune_candidates(self, keep_per_label: int) -> list[sqlite3.Row]:
        """Events whose snapshot may be deleted by the nightly retention pass.

        Only auto-confirmed / auto-ignored events with *no* human correction are
        capped (newest `keep_per_label` per label). Human corrections and false
        positives are ground truth and are never pruned.
        """
        with self.connect() as conn:
            return conn.execute(
                """
                WITH pool AS (
                    SELECT e.id, e.label AS bucket_label, e.snapshot_path, e.start_time
                    FROM events e
                    WHERE e.snapshot_path != ''
                      AND e.status IN ('confirmed', 'ignored')
                      AND NOT EXISTS (SELECT 1 FROM corrections c WHERE c.event_id = e.id)
                ),
                ranked AS (
                    SELECT id, 'shared' AS bucket, bucket_label, ROW_NUMBER() OVER (
                        PARTITION BY bucket_label ORDER BY start_time DESC
                    ) AS rn
                    FROM pool
                )
                SELECT r.id, r.bucket, r.bucket_label, p.snapshot_path, r.rn
                FROM ranked r
                JOIN events p ON p.id = r.id
                WHERE r.rn > ?
                """,
                (keep_per_label,),
            ).fetchall()

    # ---- brand review items -------------------------------------------------

    def insert_brand(self, event_id: str, brand: str, source: str = "frigate",
                     model: str | None = None, score: float | None = None) -> None:
        if not event_id or not brand:
            return
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO event_brands
                    (event_id, brand, source, model, score, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (event_id, brand, source, model, score, time.time()),
            )

    def brands_for_event(self, event_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM event_brands WHERE event_id = ? ORDER BY brand", (event_id,)
            ).fetchall()

    def brand_queue(self, status: str = "pending", brand: str | None = None,
                    camera: str | None = None, limit: int = 200) -> list[sqlite3.Row]:
        where = []
        params: list = []
        if status and status != "all":
            where.append("eb.status = ?")
            params.append(status)
        if brand:
            where.append("eb.brand = ?")
            params.append(brand)
        if camera:
            where.append("e.camera = ?")
            params.append(camera)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            "SELECT eb.*, e.camera, e.label AS object_label, e.box, e.snapshot_path, "
            "e.confidence, e.start_time, e.description "
            "FROM event_brands eb JOIN events e ON e.id = eb.event_id"
            + clause
            + " ORDER BY eb.created_at DESC LIMIT ?"
        )
        params.append(limit)
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def brand_count(self, status: str = "pending") -> int:
        with self.connect() as conn:
            if status == "all":
                row = conn.execute("SELECT COUNT(*) AS n FROM event_brands").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM event_brands WHERE status = ?", (status,)
                ).fetchone()
        return row["n"]

    def brands_by_status(self) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM event_brands GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def brands_by_brand(self, status: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as conn:
            if status:
                return conn.execute(
                    "SELECT brand, COUNT(*) AS n FROM event_brands WHERE status = ? "
                    "GROUP BY brand ORDER BY n DESC",
                    (status,),
                ).fetchall()
            return conn.execute(
                "SELECT brand, COUNT(*) AS n FROM event_brands GROUP BY brand ORDER BY n DESC"
            ).fetchall()

    def set_brand_status(self, brand_id: int, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE event_brands SET status = ?, reviewed_at = ? WHERE id = ?",
                (status, time.time(), brand_id),
            )

    def delete_brands_for_event(self, event_id: str) -> int:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM event_brands WHERE event_id = ?", (event_id,))
            return cur.rowcount

    def delete_brands_for_events(self, ids: list[str]) -> None:
        if not ids:
            return
        with self.connect() as conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM event_brands WHERE event_id IN ({marks})", chunk)

    def delete_all_brands(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM event_brands")

    # ---- per-label controls -------------------------------------------------

    def all_label_settings(self) -> dict[str, dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM label_settings").fetchall()
        return {row["label"]: dict(row) for row in rows}

    def label_settings(self, label: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM label_settings WHERE label = ?", (label,)
            ).fetchone()
        return dict(row) if row else None

    def set_label_settings(self, label: str, **fields) -> None:
        allowed = ("collect_mode", "search", "include_training", "auto_confirm", "pseudo_labels")
        row = self.label_settings(label) or {"label": label}
        values = {k: row.get(k) for k in allowed}
        for key, value in fields.items():
            if key in allowed:
                if isinstance(value, bool):
                    value = int(value)
                values[key] = value
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO label_settings
                    (label, collect_mode, search, include_training, auto_confirm, pseudo_labels, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(label) DO UPDATE SET
                    collect_mode = excluded.collect_mode,
                    search = excluded.search,
                    include_training = excluded.include_training,
                    auto_confirm = excluded.auto_confirm,
                    pseudo_labels = excluded.pseudo_labels,
                    updated_at = excluded.updated_at
                """,
                (
                    label,
                    values.get("collect_mode") or "all",
                    int(values.get("search") or 0),
                    int(values.get("include_training") if values.get("include_training") is not None else 1),
                    int(values.get("auto_confirm") if values.get("auto_confirm") is not None else 1),
                    int(values.get("pseudo_labels") if values.get("pseudo_labels") is not None else 1),
                    time.time(),
                ),
            )

    def bulk_set_label_settings(self, labels: list[str], field: str, value) -> None:
        for label in labels:
            self.set_label_settings(label, **{field: value})

    def delete_all_label_settings(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM label_settings")

    # ---- base models --------------------------------------------------------

    def base_model_insert(self, name: str, source: str, fmt: str, path: str | None,
                          imgsz: int | None = None, labels: str | None = None,
                          trainable: bool = True, meta: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO base_models (name, source, format, path, imgsz, labels, trainable, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    source = excluded.source, format = excluded.format, path = excluded.path,
                    imgsz = excluded.imgsz, labels = excluded.labels,
                    trainable = excluded.trainable, meta = excluded.meta
                """,
                (name, source, fmt, path, imgsz, labels, int(bool(trainable)), meta, time.time()),
            )

    def base_models(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM base_models ORDER BY source, name").fetchall()

    def base_model_get(self, name: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM base_models WHERE name = ?", (name,)
            ).fetchone()

    def base_model_delete(self, name: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM base_models WHERE name = ?", (name,))

    # ---- reset --------------------------------------------------------------

    def reset_training_data(self) -> None:
        """Wipe all learned/review data. Keeps base_models only."""
        with self.connect() as conn:
            for table in (
                "event_brands", "corrections", "events", "model_versions",
                "frigate_backups", "label_settings",
            ):
                conn.execute(f"DELETE FROM {table}")

    def pending_total(self) -> int:
        return self.queue_count("pending") + self.brand_count("pending")

    def kv_get(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def kv_delete(self, key: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM config WHERE key = ?", (key,))
