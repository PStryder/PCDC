"""Standalone SQLite telemetry database for per-turn deviation data."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class TelemetryDB:
    """Records per-turn steering metrics to a local SQLite database.

    All writes are fire-and-forget — errors are logged but never raised,
    so telemetry can never break generation.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.session_id = str(uuid.uuid4())
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()
        logger.info(
            "TelemetryDB opened: %s (session %s)", db_path, self.session_id,
        )

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                turn_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id          TEXT NOT NULL,
                timestamp           REAL NOT NULL,
                energy_recon        REAL NOT NULL,
                energy_predict      REAL NOT NULL,
                energy_blended      REAL NOT NULL,
                cosine_distance     REAL,
                deviation_norm      REAL NOT NULL,
                deviation_vector    BLOB NOT NULL,
                top_match_score     REAL,
                top_match_idx       INTEGER,
                adjusted_temp       REAL NOT NULL,
                converged           BOOLEAN NOT NULL,
                settle_steps        INTEGER NOT NULL,
                retrieval_triggered BOOLEAN NOT NULL
            )
        """)
        self._conn.commit()

    def record_turn(
        self,
        *,
        energy_recon: float,
        energy_predict: float,
        energy_blended: float,
        cosine_distance: float | None,
        deviation_norm: float,
        deviation_vector: bytes,
        top_match_score: float | None,
        top_match_idx: int | None,
        adjusted_temp: float,
        converged: bool,
        settle_steps: int,
        retrieval_triggered: bool,
    ) -> None:
        """Insert a turn row. Fire-and-forget — logs errors, never raises."""
        try:
            self._conn.execute(
                """
                INSERT INTO turns (
                    session_id, timestamp,
                    energy_recon, energy_predict, energy_blended,
                    cosine_distance, deviation_norm, deviation_vector,
                    top_match_score, top_match_idx,
                    adjusted_temp, converged, settle_steps,
                    retrieval_triggered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id, time.time(),
                    energy_recon, energy_predict, energy_blended,
                    cosine_distance, deviation_norm, deviation_vector,
                    top_match_score, top_match_idx,
                    adjusted_temp, converged, settle_steps,
                    retrieval_triggered,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("TelemetryDB.record_turn failed")

    def query_session(self, session_id: str) -> list[dict]:
        """Return all turns for a given session as a list of dicts."""
        cur = self._conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_id",
            (session_id,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def query_recent(self, n: int = 100) -> list[dict]:
        """Return the last *n* turns across all sessions."""
        cur = self._conn.execute(
            "SELECT * FROM turns ORDER BY turn_id DESC LIMIT ?", (n,),
        )
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
            logger.info("TelemetryDB closed: %s", self.db_path)
        except Exception:
            logger.exception("TelemetryDB.close failed")
