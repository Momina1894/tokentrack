"""
Local-only storage: one SQLite file on your machine.

Why SQLite and not JSONL? Because you're going to query this with SQL
immediately, and SQLite is a real database that pandas can read in one line:

    import sqlite3, pandas as pd
    df = pd.read_sql("select * from events", sqlite3.connect("tokentrack.db"))

Nothing ever leaves the machine. Later, the "collector" server will be a
second storage backend with the same write_event() interface.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

DEFAULT_PATH = os.environ.get("TOKENTRACK_DB", "tokentrack.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,   -- ISO-8601 UTC
    provider        TEXT    NOT NULL,   -- 'openai' | 'anthropic'
    model           TEXT    NOT NULL,   -- exact model string the provider returned
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cached_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL,               -- NULL if model not in price table
    latency_ms      INTEGER,
    conversation_id TEXT,               -- group multi-turn chats; you set this
    tag             TEXT,               -- free label, e.g. 'support-bot'
    prompt_text     TEXT,               -- only stored if store_text=True
    response_text   TEXT,               -- only stored if store_text=True
    meta            TEXT                -- JSON blob for anything else
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_conv ON events(conversation_id);
"""


class SQLiteStore:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path)

    def write_event(self, event: dict) -> None:
        """
        Insert one record. Never raises — if logging fails, the user's
        program must keep working. That rule is non-negotiable for a
        library that sits inside other people's code.
        """
        try:
            row = dict(event)
            row.setdefault("ts", datetime.now(timezone.utc).isoformat())
            row["meta"] = json.dumps(row.get("meta") or {})
            cols = [
                "ts", "provider", "model", "input_tokens", "output_tokens",
                "cached_tokens", "cost_usd", "latency_ms", "conversation_id",
                "tag", "prompt_text", "response_text", "meta",
            ]
            values = [row.get(c) for c in cols]
            with self._lock, self._connect() as conn:
                conn.execute(
                    f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    values,
                )
        except Exception as e:  # noqa: BLE001
            if os.environ.get("TOKENTRACK_DEBUG"):
                print(f"[tokentrack] failed to write event: {e}")
