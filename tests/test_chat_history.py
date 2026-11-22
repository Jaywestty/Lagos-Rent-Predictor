from conversation.history import _title, history_payload, save_exchange, valid_uuid


def test_title_is_single_line_and_bounded():
    assert _title("  how much   is rent\nin Yaba ") == "how much is rent in Yaba"
    long = _title("x" * 200)
    assert len(long) == 80 and long.endswith("…")


def test_valid_uuid_rejects_garbage():
    assert valid_uuid("6f1c2d3e-0000-4000-8000-000000000001") == "6f1c2d3e-0000-4000-8000-000000000001"
    assert valid_uuid("not-a-uuid") is None
    assert valid_uuid(None) is None


def test_history_payload_keeps_answer_and_unique_listings():
    payload = {
        "query_type": "affordability",
        "answer": {"status": "success", "median": 1},
        "areas": [{"results": [{"url": "u1"}, {"url": "u2"}]}, {"results": [{"url": "u1"}]}],
    }
    out = history_payload(payload)
    assert out["answer"]["median"] == 1
    assert [x["url"] for x in out["listings"]] == ["u1", "u2"]
    assert "advice" not in out


def test_history_payload_keeps_advice_with_sources():
    payload = {"query_type": "advice", "answer": {"message": "Ask for a receipt."}, "sources": [{"title": "t", "url": "s"}]}
    out = history_payload(payload)
    assert out["advice"] == {"text": "Ask for a receipt.", "sources": [{"title": "t", "url": "s"}]}


def test_nothing_is_saved_without_valid_ids():
    # Returns before touching the database.
    assert save_exchange(None, "6f1c2d3e-0000-4000-8000-000000000001", "q", {}) is False
    assert save_exchange("6f1c2d3e-0000-4000-8000-000000000001", "bad", "q", {}) is False
    assert save_exchange("6f1c2d3e-0000-4000-8000-000000000001", "6f1c2d3e-0000-4000-8000-000000000002", "  ", {}) is False


def test_thread_endpoints_require_a_device_id():
    from fastapi.testclient import TestClient

    from entrypoints.api import app

    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/api/me/threads").status_code == 400
    assert c.get("/api/me/threads/abc", headers={"X-Client-Id": "6f1c2d3e-0000-4000-8000-000000000001"}).status_code == 404
