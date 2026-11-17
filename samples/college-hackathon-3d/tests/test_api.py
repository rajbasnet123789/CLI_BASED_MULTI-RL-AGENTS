"""API smoke tests for the HackVerse backend."""

from __future__ import annotations

import pytest

from server import app, create_app


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert "HackVerse" in body["event"]


def test_tracks(client) -> None:
    response = client.get("/api/tracks")
    assert response.status_code == 200
    tracks = response.get_json()
    assert len(tracks) == 4
    assert {track["id"] for track in tracks} == {"ai", "web", "games", "hardware"}


def test_schedule(client) -> None:
    response = client.get("/api/schedule")
    assert response.status_code == 200
    items = response.get_json()
    assert items
    assert all({"time", "title", "day"} <= set(item) for item in items)


def test_teams(client) -> None:
    response = client.get("/api/teams")
    assert response.status_code == 200
    teams = response.get_json()
    assert isinstance(teams, list)


def test_stats(client) -> None:
    response = client.get("/api/stats")
    assert response.status_code == 200
    stats = response.get_json()
    assert "teams" in stats
    assert "tracks" in stats
    assert "members" in stats
    
    # Verify values match the current state of TEAMS and TRACKS
    teams_list = client.get("/api/teams").get_json()
    tracks_list = client.get("/api/tracks").get_json()
    expected_teams = len(teams_list)
    expected_tracks = len(tracks_list)
    expected_members = sum(t["members"] for t in teams_list)
    
    assert stats["teams"] == expected_teams
    assert stats["tracks"] == expected_tracks
    assert stats["members"] == expected_members


def test_register_valid(client) -> None:
    before = len(client.get("/api/teams").get_json())
    response = client.post(
        "/api/register",
        json={"name": "Test Crew", "track": "ai", "members": 2},
    )
    assert response.status_code == 201
    team = response.get_json()
    assert team["name"] == "Test Crew"
    assert team["track"] == "ai"
    after = len(client.get("/api/teams").get_json())
    assert after == before + 1


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "track": "ai", "members": 2},
        {"name": "Team", "track": "unknown", "members": 2},
        {"name": "Team", "track": "ai", "members": 0},
        {"name": "Team", "track": "ai", "members": 9},
    ],
)
def test_register_invalid(client, payload) -> None:
    response = client.post("/api/register", json=payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_index_page(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"HackVerse" in response.data


def test_create_app_isolation() -> None:
    assert create_app().test_client().get("/api/health").status_code == 200
