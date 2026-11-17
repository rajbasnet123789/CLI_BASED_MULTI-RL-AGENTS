"""Simple in-memory backend for the HackVerse 2026 landing page."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

EVENT_NAME = "HackVerse 2026"
STATIC_DIR = Path(__file__).resolve().parent / "static"

TRACKS = [
    {"id": "ai", "name": "AI & Machine Learning", "icon": "brain", "color": "#7c5cff"},
    {"id": "web", "name": "Web & Mobile", "icon": "globe", "color": "#00d4ff"},
    {"id": "games", "name": "Games & 3D", "icon": "cube", "color": "#ff4fa3"},
    {"id": "hardware", "name": "Hardware & IoT", "icon": "chip", "color": "#ffb347"},
]

SCHEDULE = [
    {"time": "09:00", "title": "Check-in & breakfast", "day": 1},
    {"time": "10:00", "title": "Opening ceremony", "day": 1},
    {"time": "11:00", "title": "Hacking begins", "day": 1},
    {"time": "13:00", "title": "Mentor office hours", "day": 1},
    {"time": "09:00", "title": "Final submissions", "day": 2},
    {"time": "11:00", "title": "Demos & judging", "day": 2},
    {"time": "14:00", "title": "Awards & closing", "day": 2},
]

TEAMS: list[dict[str, Any]] = [
    {"id": 1, "name": "Null Pointers", "track": "web", "members": 4},
    {"id": 2, "name": "Quantum Cats", "track": "ai", "members": 3},
    {"id": 3, "name": "Pixel Pirates", "track": "games", "members": 5},
]

_ids = itertools.count(max((team["id"] for team in TEAMS), default=0) + 1)


def create_app() -> Flask:
    """Build the Flask application."""
    app = Flask(__name__, static_folder=None)

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "event": EVENT_NAME})

    @app.get("/api/tracks")
    def tracks() -> Any:
        return jsonify(TRACKS)

    @app.get("/api/schedule")
    def schedule() -> Any:
        return jsonify(SCHEDULE)

    @app.get("/api/teams")
    def teams() -> Any:
        return jsonify(TEAMS)

    @app.get("/api/stats")
    def stats() -> Any:
        return jsonify({
            "teams": len(TEAMS),
            "tracks": len(TRACKS),
            "members": sum(t["members"] for t in TEAMS)
        })

    @app.post("/api/register")
    def register() -> Any:
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip()
        track = str(payload.get("track", "")).strip()
        members = payload.get("members", 1)

        if not name:
            return jsonify({"error": "name is required"}), 400
        if track not in {item["id"] for item in TRACKS}:
            return jsonify({"error": "unknown track"}), 400
        if not isinstance(members, int) or not 1 <= members <= 6:
            return jsonify({"error": "members must be between 1 and 6"}), 400

        team = {"id": next(_ids), "name": name, "track": track, "members": members}
        TEAMS.append(team)
        return jsonify(team), 201

    @app.get("/")
    def index() -> Any:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str) -> Any:
        return send_from_directory(STATIC_DIR, filename)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
