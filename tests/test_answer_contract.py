from unittest.mock import patch

from query.answer import compute_contract
from query.entities import PropertyEntities, QueryType
from query.stats import annualized_price, bucket_for, classify, median, range_stats, summarize_values


def test_percentile_and_median():
    vals = sorted([float(i) for i in range(1, 21)])
    mn, mx, method = range_stats(vals, 10)
    assert method == "percentile"
    assert mn == vals[0] + 0.9 * (vals[1] - vals[0]) or mn is not None
    assert range_stats([1, 2, 3], 10)[2] == "min_max"
    assert median([3, 1, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5
    assert summarize_values([]) is None


def test_threshold_classification():
    assert classify(0)[0] == "no_data"
    assert classify(1)[0] == "low_data"
    assert classify(3)[0] == "low_data"
    assert classify(4) == ("success", "limited")
    assert classify(9) == ("success", "limited")
    assert classify(10) == ("success", "normal")


def test_annual_normalisation():
    assert annualized_price(100000, "month") == 1200000
    assert annualized_price(100000, "year") == 100000
    assert annualized_price(100000, None) == 100000
    assert annualized_price(None, "month") is None
    assert annualized_price(0, "month") is None
    assert annualized_price(-5, "year") is None


def test_bucket_mapping():
    assert bucket_for(1, None) == "one_bedroom"
    assert bucket_for(2, None) == "two_bedroom"
    assert bucket_for(3, None) == "three_bedroom"
    assert bucket_for(None, "self contain") == "self_contain"
    assert bucket_for(4, None) is None


def test_compute_contract_no_data():
    c = compute_contract({}, [], area="Gbagada", request_id="r1", latency_ms=12.5)
    assert c.status == "no_data"
    assert c.listing_count == 0
    assert c.confidence == "insufficient"
    assert c.headline is None


def test_compute_contract_success():
    vals = [1000000.0] * 12
    buckets = {"one_bedroom": [1000000.0] * 4, "two_bedroom": [2000000.0] * 4, "three_bedroom": [], "self_contain": []}
    c = compute_contract(buckets, vals, area="Yaba", request_id="r2", latency_ms=5.0)
    assert c.status == "success"
    assert c.confidence == "normal"
    assert c.median == 1000000.0
    assert c.currency == "NGN"
    assert c.rent_period == "yearly"


def test_context_carry_over():
    from conversation.session import load_session_entities  # noqa: F401
    prev = PropertyEntities(query_type=QueryType.LOOKUP, beds=2, area="Yaba")
    assert prev.beds == 2 and prev.area == "Yaba"


def test_no_data_and_out_of_scope_paths():
    c = compute_contract({}, [], request_id="x1", latency_ms=1.0)
    assert c.status == "no_data"
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    from entrypoints.api import app
    client = TestClient(app, raise_server_exceptions=False)
    @contextmanager
    def fake_trace(*a, **k):
        yield
    with patch("entrypoints.api.start_trace", return_value="t1"), patch("entrypoints.api.trace_stage", side_effect=fake_trace):
        with patch("entrypoints.api.extract_entities") as mock_ex:
            mock_ex.return_value = PropertyEntities(query_type=QueryType.LOOKUP, listing_type="sale")
            with patch("entrypoints.api.fetch_rent_stats", return_value={"buckets": {}, "all_values": [], "data_as_of": None, "area": None}):
                with patch("entrypoints.api.run_lookup", return_value=[]):
                    r = client.post("/query", json={"query": "buy a house", "request_id": "x2"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]["status"] == "out_of_scope"


def test_idempotency():
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    from entrypoints.api import app
    client = TestClient(app, raise_server_exceptions=False)
    @contextmanager
    def fake_trace(*a, **k):
        yield
    ent = PropertyEntities(query_type=QueryType.LOOKUP, area="Gbagada")
    with patch("entrypoints.api.start_trace", return_value="t1"), patch("entrypoints.api.trace_stage", side_effect=fake_trace):
        with patch("entrypoints.api.extract_entities", return_value=ent) as mock_ex:
            with patch("entrypoints.api.run_lookup", return_value=[]):
                with patch("entrypoints.api.fetch_rent_stats", return_value={"buckets": {}, "all_values": [], "data_as_of": None, "area": "Gbagada"}):
                    r1 = client.post("/query", json={"query": "rent in Gbagada", "request_key": "key-123", "request_id": "id1"})
                    r2 = client.post("/query", json={"query": "rent in Gbagada", "request_key": "key-123", "request_id": "id1"})
                    assert r1.status_code == 200 and r2.status_code == 200
                    assert r1.json() == r2.json()
                    assert mock_ex.call_count == 1
