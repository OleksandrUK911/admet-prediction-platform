"""Profile storage, isolated from FastAPI (same pattern as inference.py).

SQLite is enough for this project's scale (a portfolio demo, not a
production service) - the full profile response is stored as JSON per row
so GET /admet-profile/{id} can just look it up and GET /history can list
recent ones without re-deriving anything.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "app.db"


@contextmanager
def get_connection():
    # sqlite3's own context manager only commits/rolls back on exit, it
    # does NOT close the connection - wrap it so callers can't leak file
    # handles (ResourceWarning: unclosed database, same fix as project #1).
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                smiles TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Speeds up list_history()'s ORDER BY created_at DESC.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_profiles_created_at ON profiles(created_at)")


def insert_profile(id: str, smiles: str, profile_json: dict, created_at: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO profiles (id, smiles, profile_json, created_at) VALUES (?, ?, ?, ?)",
            (id, smiles, json.dumps(profile_json), created_at),
        )


def get_profile(id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT profile_json FROM profiles WHERE id = ?", (id,)).fetchone()
    if row is None:
        return None
    return json.loads(row["profile_json"])


def list_history(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, smiles, created_at FROM profiles ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
