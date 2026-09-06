from pathlib import Path

from fastapi.testclient import TestClient

from marg.api.app import create_app


def test_read_only_upload_and_health(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MARG_READ_ONLY", "1")
    client = TestClient(create_app(tmp_path))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["read_only"] is True

    response = client.post(
        "/api/surveys/upload",
        files={"video": ("clip.mp4", b"video", "video/mp4")},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Uploads are disabled on the public demo; run locally or on AWS"
    )
