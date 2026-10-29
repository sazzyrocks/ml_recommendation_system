"""
tests/test_health.py
Smoke-test: verifies the /health endpoint is reachable.
No DB connection required — the endpoint is intentionally dependency-free.
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "environment" in data
