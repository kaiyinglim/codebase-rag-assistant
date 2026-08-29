"""Tests for the FastAPI endpoint contracts."""

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_health_returns_200():
    """Checks that the health endpoint is available."""
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_query_rejects_empty_question():
    """Checks that whitespace-only questions are rejected."""
    response = client.post(
        "/query",
        json={
            "question": "   ",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_query_rejects_invalid_top_k():
    """Checks that top_k cannot be less than one."""
    response = client.post(
        "/query",
        json={
            "question": "How is authentication handled?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_query_returns_answer_and_sources(monkeypatch):
    """Checks that a successful query follows the expected response format."""

    # Replace the real RAG pipeline so this test does not call DeepSeek.
    def fake_answer_question(question: str, top_k: int):
        return {
            "answer": "Authentication is handled in auth.py.",
            "sources": [
                {
                    "file": "auth.py",
                    "name": "authenticate",
                    "start_line": 10,
                    "end_line": 20,
                }
            ],
        }

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/query",
        json={
            "question": "How is authentication handled?",
            "top_k": 3,
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == (
        "Authentication is handled in auth.py."
    )
    assert len(body["sources"]) == 1
    assert body["sources"][0]["file"] == "auth.py"


def test_index_rejects_invalid_repo_path():
    """Checks that indexing a missing repository returns a clear error."""
    response = client.post(
        "/index",
        json={
            "repo_path": "/this/repository/does/not/exist",
        },
    )

    assert response.status_code == 400
    assert "Repository path does not exist" in response.json()["detail"]