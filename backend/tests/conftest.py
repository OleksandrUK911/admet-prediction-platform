import pytest

import backend.app.main as main_module
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


@pytest.fixture(autouse=True)
def reset_prediction_cache():
    """ModelService.predict()'s in-memory cache lives on the module-level
    model_service instance, not per-request state. Most tests already get a
    fresh instance per `with TestClient(app)` block (lifespan reloads the
    model), but a module-scoped fixture (e.g. test_inference.py's `service`)
    or a test running outside a fresh TestClient context could otherwise see
    another test's cached predictions - clear it explicitly either way."""
    if main_module.model_service is not None:
        main_module.model_service.clear_cache()
    yield
    if main_module.model_service is not None:
        main_module.model_service.clear_cache()
