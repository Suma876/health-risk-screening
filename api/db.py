"""SQLite storage with hand-written SQL (no ORM)."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    input_json  TEXT    NOT NULL,
    risk        REAL    NOT NULL,
    band        TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL
)
"""


def _connect():
    con = sqlite3.connect(os.getenv("DB_PATH", "requests.db"), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")   # readers don't block the writer
    con.execute(SCHEMA)
    return con


def save_request(payload: dict, risk: float, band: str, latency_ms: float):
    with closing(_connect()) as con:
        con.execute(
            "INSERT INTO requests (created_at, input_json, risk, band, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(payload),
             risk, band, latency_ms),
        )
        con.commit()


def get_stats() -> dict:
    with closing(_connect()) as con:
        total, avg_risk, high, avg_lat = con.execute(
            """SELECT COUNT(*),
                      AVG(risk),
                      SUM(CASE WHEN band = 'high' THEN 1 ELSE 0 END),
                      AVG(latency_ms)
               FROM requests"""
        ).fetchone()
    total = total or 0
    return {
        "total_requests": total,
        "average_risk": round(avg_risk, 4) if total else 0.0,
        "high_risk_share": round((high or 0) / total, 4) if total else 0.0,
        "avg_latency_ms": round(avg_lat, 2) if total else 0.0,
    }
