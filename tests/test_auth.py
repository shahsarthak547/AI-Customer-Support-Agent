
from fastapi.testclient import TestClient

from app.api.main import app

TEST_API_KEY = "test-secret"


def test_missing_api_key_returns_401(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    with TestClient(app) as client:
        response = client.get("/api/v1/intents")

    assert response.status_code == 401


def test_invalid_api_key_returns_401(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/intents",
            headers={"X-API-Key": "wrong-secret"},
        )

    assert response.status_code == 401


def test_valid_api_key_allows_request(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/intents",
            headers={"X-API-Key": TEST_API_KEY},
        )

    assert response.status_code == 200


def test_missing_server_configuration_returns_503(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/intents")

    assert response.status_code == 503


def test_health_endpoint_remains_public(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
