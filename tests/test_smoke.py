"""
Smoke tests for the CI pipeline.

These deliberately touch only the two endpoints that work with no
GROQ_API_KEY / HF_TOKEN and make no network/LLM calls (/api/health,
/api/config) — the goal is just to catch "the app doesn't even import
or boot" type breakage (a bad import, a typo in a route decorator, a
StaticFiles path that doesn't exist, etc.) before it reaches Docker
build or deploy. This is NOT a replacement for testing the actual
RAG/ingest/ask logic, which needs real API keys and network access.
"""

from fastapi.testclient import TestClient

import server


client = TestClient(server.app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_config_endpoint_shape():
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert "has_server_key" in body
    assert isinstance(body["has_server_key"], bool)


def test_documents_endpoint_starts_empty():
    # A fresh app instance (no prior /api/ingest/* call in this process)
    # should report no filed sources yet.
    res = client.get("/api/documents")
    assert res.status_code == 200
    assert res.json() == {"sources": []}


def test_ask_with_empty_question_is_rejected():
    res = client.post("/api/ask", json={"question": "   "})
    assert res.status_code == 400


def test_frontend_is_served():
    # index.html should be served at "/" — catches a broken STATIC_DIR path.
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
