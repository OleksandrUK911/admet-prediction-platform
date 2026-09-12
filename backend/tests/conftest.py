import pytest

from backend.app import db
from backend.app.main import limiter


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    """Every test gets its own throwaway SQLite file - never touches the
    real app.db, and tests can't see each other's rows."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """The rate limiter's in-memory counters are a module-level global, not
    per-request state - without resetting, one test that makes many
    /admet-profile calls (e.g. the 429 test) would trip the limit for every
    other test sharing the same TestClient "IP" in the same process."""
    limiter.reset()
    yield
    limiter.reset()
