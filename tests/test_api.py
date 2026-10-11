#!/usr/bin/env python3
"""Fixed API integration test: JWT + correct routes + status polling.

Replaces stale root test.py (which used /upload and /ask with no auth).

Usage:
    .venv/bin/python -m pytest tests/test_api.py -v -s
    # or directly (needs running stack):
    EVAL_EMAIL=t@t.com EVAL_PASSWORD=secret123 .venv/bin/python tests/test_api.py

Requires: docker-compose up -d, uvicorn app.main:app, celery worker.
"""
import os
import sys
import time

import requests

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
EMAIL = os.getenv("EVAL_EMAIL", "apitest@example.com")
PASSWORD = os.getenv("EVAL_PASSWORD", "apitest123")


def get_token() -> str:
    r = requests.post(f"{BASE_URL}/auth/login",
                      data={"username": EMAIL, "password": PASSWORD})
    if r.status_code != 200:
        reg = requests.post(f"{BASE_URL}/auth/register",
                            json={"email": EMAIL, "password": PASSWORD})
        assert reg.status_code in (200, 201, 400), reg.text
        r = requests.post(f"{BASE_URL}/auth/login",
                          data={"username": EMAIL, "password": PASSWORD})
    assert r.status_code == 200, f"login failed: {r.text}"
    return r.json()["access_token"]


def wait_for_status(token: str, doc_id: str, timeout: int = 180) -> dict:
    h = {"Authorization": f"Bearer {token}"}
    for _ in range(timeout // 3):
        r = requests.get(f"{BASE_URL}/documents/{doc_id}", headers=h)
        assert r.status_code == 200, r.text
        doc = r.json()
        if doc["status"] in ("COMPLETED", "FAILED"):
            return doc
        time.sleep(3)
    raise TimeoutError(f"doc {doc_id} not done after {timeout}s")


def test_full_pipeline():
    """Upload synthetic PDF -> COMPLETED -> ask -> debug. Skips if server down."""
    try:
        requests.get(f"{BASE_URL}/", timeout=5)
    except requests.ConnectionError:
        print(f"SKIP: server not up at {BASE_URL}")
        return
    token = get_token()
    h = {"Authorization": f"Bearer {token}"}
    # resolve corpus path robustly
    corpus = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "eval", "synthetic_corpus", "company_policy.pdf")
    assert os.path.exists(corpus), f"run eval/generate_corpus.py first ({corpus} missing)"
    with open(corpus, "rb") as f:
        r = requests.post(f"{BASE_URL}/documents/upload", headers=h,
                          files={"file": ("company_policy.pdf", f, "application/pdf")})
    assert r.status_code in (200, 201), r.text
    doc_id = r.json()["id"]
    doc = wait_for_status(token, doc_id)
    assert doc["status"] == "COMPLETED", f"ingestion failed: {doc.get('error_log')}"
    q = "What is the Project Phoenix budget?"
    ans = requests.get(f"{BASE_URL}/query/ask", headers=h, params={"question": q})
    assert ans.status_code == 200, ans.text
    assert "42,500" in ans.json().get("answer", ""), ans.json()
    dbg = requests.get(f"{BASE_URL}/query/debug", headers=h, params={"question": q})
    assert dbg.status_code == 200 and len(dbg.json()) > 0


if __name__ == "__main__":
    test_full_pipeline()
    print("API pipeline test passed.")
