import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from marg.api.app import create_app


@pytest.fixture
def setup(monkeypatch, tmp_path):
    seeds = tmp_path / "seeds"
    seed = seeds / "pothole_kumasi"
    seed.mkdir(parents=True)
    (seed / "result.json").write_text('{"original":true}')
    monkeypatch.setenv("MARG_READ_ONLY", "1")
    monkeypatch.setenv("MARG_LIVE_DEMO_ENABLED", "1")
    monkeypatch.setenv("MARG_LIVE_DATA_ROOT", str(tmp_path / "live"))
    monkeypatch.setenv("MARG_MODAL_ALLOWANCE_DB", str(tmp_path / "budget.sqlite3"))
    monkeypatch.setenv("MARG_MODAL_ENABLED", "1")
    monkeypatch.setenv("MARG_MODAL_PROXY_TOKEN", "local-test-only")
    monkeypatch.setenv("MARG_COOKIE_SECURE", "0")

    def run(directory, provider):
        assert provider == "modal"
        (directory / "result.json").write_text("Changed private copy")
        return {
            "status": "completed",
            "run_id": "run-live",
            "audit": {"passed": True},
            "provider": {"kind": "modal", "live_inference": True},
            "work_orders": [{"work_order_id": "order-live"}],
        }

    monkeypatch.setattr("marg.api.app._run_agent", run)
    app = create_app(seeds)
    return app, seed


def completed(client, job):
    for _ in range(100):
        value = client.get(job["status_url"]).json()
        if value["status"] not in {"queued", "running"}:
            return value
        time.sleep(0.01)
    pytest.fail("Local test job did not complete")


def test_live_reviews_isolate_sessions_and_fixture_data(setup):
    app, seed = setup
    with TestClient(app) as client:
        response = client.post(
            "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
        )
        assert response.status_code == 202
        job = completed(client, response.json())
        assert job["status"] == "succeeded"
        assert json.loads((seed / "result.json").read_text()) == {"original": True}
        with TestClient(app) as stranger:
            assert stranger.get(job["status_url"]).status_code == 404
        assert client.get(job["status_url"]).status_code == 200
        assert client.post("/api/surveys/pothole_kumasi/run_agent").status_code == 403


def test_live_human_decision_requires_owner_and_records_no_dispatch(setup):
    app, _ = setup
    with TestClient(app) as client:
        job = completed(
            client,
            client.post(
                "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
            ).json(),
        )
        path = job["status_url"] + "/decisions/order-live"
        with TestClient(app) as stranger:
            assert stranger.post(path, json={"decision": "approve"}).status_code == 404
        result = client.post(
            path, json={"decision": "approve", "note": "Judge sample only"}
        )
        assert result.status_code == 200
        decision = client.get(job["status_url"]).json()["decisions"]["order-live"]
        assert decision["dispatch"] is False and decision["note"] == "Judge sample only"
        assert client.post(path, json={"decision": "reject"}).status_code == 409


def test_live_policy_failure_blocks_decisions(setup):
    app, _ = setup
    app.state.live_reviews.runner = lambda *args: {
        "status": "failed",
        "error": {"message": "Audit failed"},
        "audit": {"passed": False},
    }
    with TestClient(app) as client:
        job = completed(
            client,
            client.post(
                "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
            ).json(),
        )
        assert (
            client.post(
                job["status_url"] + "/decisions/order-live",
                json={"decision": "approve"},
            ).status_code
            == 409
        )


def test_live_rejects_arbitrary_input_cross_origin_and_large_requests(setup):
    app, _ = setup
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/live-reviews", json={"survey_id": "../../private"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/live-reviews",
                json={"survey_id": "pothole_kumasi", "prompt": "arbitrary"},
            ).status_code
            == 422
        )
        assert client.post("/api/live-reviews", content=b"x" * 17000).status_code == 413
        assert (
            client.post(
                "/api/live-reviews",
                json={"survey_id": "pothole_kumasi"},
                headers={"Origin": "https://foreign.example"},
            ).status_code
            == 403
        )


def test_live_rate_and_concurrency_limits(setup):
    app, _ = setup
    event = threading.Event()
    original = app.state.live_reviews.runner
    app.state.live_reviews.runner = lambda *args: event.wait(2) and original(*args)
    with TestClient(app) as client:
        first = client.post(
            "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
        ).json()
        assert (
            client.post(
                "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
            ).status_code
            == 429
        )
        event.set()
        completed(client, first)
        completed(
            client,
            client.post(
                "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
            ).json(),
        )
        assert (
            client.post(
                "/api/live-reviews", json={"survey_id": "pothole_kumasi"}
            ).status_code
            == 429
        )
