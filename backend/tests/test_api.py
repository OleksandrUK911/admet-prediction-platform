import time

from fastapi.testclient import TestClient

from backend.app.main import app

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["db_connected"] is True


def test_predict_valid_smiles():
    with TestClient(app) as client:
        response = client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
    assert response.status_code == 200
    body = response.json()
    assert body["smiles"]
    assert body["disclaimer"]
    assert body["model_version"]
    assert "id" in body
    assert set(body["profile"].keys()) == {
        "solubility", "bbb_penetration", "toxicity_tox21", "clinical_trial_toxicity", "fda_approval_likelihood",
    }
    assert len(body["profile"]["toxicity_tox21"]["panel"]) == 12
    assert "applicability_domain" in body


def test_predict_invalid_smiles_returns_422():
    with TestClient(app) as client:
        response = client.post("/admet-profile", json={"smiles": "!!!not-a-smiles!!!"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Could not parse this SMILES string"}


def test_predict_empty_smiles_rejected_by_schema():
    with TestClient(app) as client:
        response = client.post("/admet-profile", json={"smiles": ""})
    assert response.status_code == 422  # Pydantic min_length validation


def test_get_profile_by_id_round_trip():
    with TestClient(app) as client:
        create_response = client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        profile_id = create_response.json()["id"]
        get_response = client.get(f"/admet-profile/{profile_id}")
    assert get_response.status_code == 200
    assert get_response.json() == create_response.json()


def test_get_profile_unknown_id_returns_404():
    with TestClient(app) as client:
        response = client.get("/admet-profile/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_history_round_trip():
    with TestClient(app) as client:
        create_response = client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        history_response = client.get("/history")
    assert history_response.status_code == 200
    rows = history_response.json()
    assert len(rows) == 1
    assert rows[0]["id"] == create_response.json()["id"]
    assert rows[0]["smiles"]
    assert "created_at" in rows[0]


def test_history_newest_first():
    with TestClient(app) as client:
        client.post("/admet-profile", json={"smiles": "CCO"})
        client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        response = client.get("/history")
    rows = response.json()
    assert len(rows) == 2
    assert rows[0]["created_at"] >= rows[1]["created_at"]


def test_history_empty_when_no_predictions_made():
    with TestClient(app) as client:
        response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_failed_prediction_does_not_write_history():
    with TestClient(app) as client:
        client.post("/admet-profile", json={"smiles": "!!!not-a-smiles!!!"})
        response = client.get("/history")
    assert response.json() == []


def test_metrics_returns_prometheus_text_after_a_request():
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_rate_limit_returns_429_after_threshold():
    with TestClient(app) as client:
        responses = [client.post("/admet-profile", json={"smiles": "CCO"}) for _ in range(31)]
    assert responses[-1].status_code == 429
    assert "detail" in responses[-1].json()


def test_repeated_smiles_served_from_cache_faster_with_identical_prediction():
    """See TODO/backend/TODO_database.md's "Кешування" section: the second
    call for the same SMILES should hit ModelService's in-memory cache -
    same prediction values, but its own fresh id/timestamp (history must
    still gain a new row per request, cache or no cache)."""
    with TestClient(app) as client:
        start_first = time.perf_counter()
        first = client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        first_duration = time.perf_counter() - start_first

        start_second = time.perf_counter()
        second = client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        second_duration = time.perf_counter() - start_second

    assert first.status_code == 200
    assert second.status_code == 200
    first_body, second_body = first.json(), second.json()

    # Fresh identity per request, even on a cache hit.
    assert first_body["id"] != second_body["id"]

    # Same underlying prediction, served from cache.
    assert first_body["profile"] == second_body["profile"]
    assert first_body["applicability_domain"] == second_body["applicability_domain"]
    assert first_body["model_version"] == second_body["model_version"]

    # Cache hit skips descriptor/fingerprint computation and all 16 model
    # predict calls, so it should be substantially faster than the first,
    # uncached call.
    assert second_duration < first_duration


def test_cached_prediction_still_writes_a_new_history_row():
    """A cache hit must not skip the history write - every successful
    /admet-profile call persists its own row, cached prediction or not."""
    with TestClient(app) as client:
        client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        client.post("/admet-profile", json={"smiles": ASPIRIN_SMILES})
        history_response = client.get("/history")

    rows = history_response.json()
    assert len(rows) == 2
    assert rows[0]["id"] != rows[1]["id"]


def test_unhandled_exception_returns_generic_500(monkeypatch):
    import backend.app.main as main_module

    def boom(_smiles):
        raise RuntimeError("simulated failure")

    # raise_server_exceptions=False: otherwise TestClient re-raises the
    # original exception for debugging even though our handler already
    # produced a real 500 response - we want to assert on that response.
    with TestClient(app, raise_server_exceptions=False) as client:
        # Patch only after lifespan startup has run, so model_service is
        # already the real instance (not None).
        monkeypatch.setattr(main_module.model_service, "predict", boom)
        response = client.post("/admet-profile", json={"smiles": "CCO"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
