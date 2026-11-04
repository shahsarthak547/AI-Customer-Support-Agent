
from fastapi.testclient import TestClient

from app.api.main import app

TEST_API_KEY = "test-secret"

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_classify(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/classify",
            json={
                "message": "My iPhone battery is draining really quickly"
            },
            headers={"X-API-Key" : TEST_API_KEY}
        )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "battery_charging"
    assert 0 <= data["confidence"] <= 1


def test_invalid_request(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/classify",
            json={
                "message": ""
            },
            headers={"X-API-Key" : TEST_API_KEY}
        )
    assert response.status_code == 422
def test_whitespace_message(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/classify",
            json={"message": "     "}, headers={"X-API-Key" : TEST_API_KEY}
        )

    assert response.status_code == 422
def test_readiness():
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"