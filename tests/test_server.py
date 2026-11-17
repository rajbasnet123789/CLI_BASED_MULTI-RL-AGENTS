"""Optional HTTP API."""

from __future__ import annotations

import pytest

from orion.config import OrionConfig

pytest.importorskip("fastapi")


def test_health_config_and_status(config: OrionConfig) -> None:
    from fastapi.testclient import TestClient

    from orion.server.app import create_app

    client = TestClient(create_app(config))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["provider"] == config.llm_provider
    assert health.json()["model"] == config.active_model
    assert health.json()["api_key"] is False

    described = client.get("/config")
    assert described.status_code == 200
    assert described.json()["embeddings"] == "local"

    status = client.get("/status")
    assert status.status_code == 200
    assert "index" in status.json()


def test_search_endpoint(config: OrionConfig) -> None:
    from fastapi.testclient import TestClient

    from orion.server.app import create_app

    client = TestClient(create_app(config))
    response = client.get("/search", params={"q": "database", "k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "database"
    assert isinstance(payload["hits"], list)

    bad = client.get("/search", params={"q": ""})
    assert bad.status_code == 422
