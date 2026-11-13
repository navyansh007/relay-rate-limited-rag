import json

from fastapi.testclient import TestClient

from relay.trace import Trace
from relay.ui.server import create_app


def client():
    return TestClient(create_app(real=False))


def test_pages_and_demo_data_are_served():
    c = client()
    assert c.get("/").status_code == 200
    demo = c.get("/static/demo.json").json()
    assert demo["sessions"] and demo["burst"]["lanes"]
    # every replayed session is a real run with spans and passages
    for s in demo["sessions"]:
        assert s["source"] == "real Groq" and s["spans"] and s["passages"]


def test_status_explains_why_live_is_off():
    st = client().get("/api/status").json()
    assert set(st) == {"live", "source", "reason"}
    if not st["live"]:
        assert st["reason"]


def test_ask_rejects_empty_and_oversized_questions():
    c = client()
    assert c.post("/api/ask", json={"question": "  "}).status_code == 400
    assert c.post("/api/ask", json={"question": "x" * 401}).status_code == 400


def test_trace_reports_spans_as_they_are_recorded():
    seen = []
    t = Trace("r1")
    t.on_span = seen.append
    t.add("work", "retrieve.0", 0, 12.5, hits=["a"])
    assert [s.name for s in seen] == ["retrieve.0"]
    assert json.loads(json.dumps(t.to_dict()))["spans"][0]["attrs"]["hits"] == ["a"]
