from fastapi.testclient import TestClient

from entrypoints.api import app


def test_root_says_what_this_address_is():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "Adugbo AI API"
    assert r.json()["health"] == "/health"


def test_health_answers_head_requests_for_uptime_pingers():
    client = TestClient(app)
    assert client.head("/health").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
