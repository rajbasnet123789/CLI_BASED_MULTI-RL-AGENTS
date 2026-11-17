from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import pytest

from backend.main import app, get_db
from backend.database import Base
from backend.models import Project, Task

# Setup for testing with an in-memory SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_db():
    """Give every test a fresh set of tables."""
    db = TestingSessionLocal()
    try:
        db.query(Task).delete()
        db.query(Project).delete()
        db.commit()
    finally:
        db.close()
    yield


def test_create_project():
    response = client.post(
        "/projects/",
        json={"name": "Test Project"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Project"
    assert "id" in data

def test_read_projects():
    response = client.get("/projects/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_read_project():
    # First create a project
    project_response = client.post(
        "/projects/",
        json={"name": "Project to Read"},
    )
    project_id = project_response.json()["id"]

    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Project to Read"
    assert data["id"] == project_id

def test_read_nonexistent_project():
    response = client.get("/projects/9999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Project not found"}

def test_update_project():
    # First create a project
    project_response = client.post(
        "/projects/",
        json={"name": "Project to Update"},
    )
    project_id = project_response.json()["id"]

    response = client.put(
        f"/projects/{project_id}",
        json={"name": "Updated Project Name"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Project Name"
    assert data["id"] == project_id

def test_delete_project():
    # First create a project
    project_response = client.post(
        "/projects/",
        json={"name": "Project to Delete"},
    )
    project_id = project_response.json()["id"]

    response = client.delete(f"/projects/{project_id}")
    assert response.status_code == 200
    assert response.json() == {"message": "Project deleted successfully"}

    # Verify it's deleted
    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 404

def test_create_task():
    # First create a project
    project_response = client.post(
        "/projects/",
        json={"name": "Task Project"},
    )
    project_id = project_response.json()["id"]

    response = client.post(
        "/tasks/",
        json={
            "title": "Test Task",
            "description": "This is a test task.",
            "status": "pending",
            "project_id": project_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Task"
    assert data["project_id"] == project_id
    assert "id" in data

def test_read_tasks():
    response = client.get("/tasks/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_read_tasks_by_project_id():
    # Create two projects
    project1_response = client.post("/projects/", json={"name": "Project One"})
    project1_id = project1_response.json()["id"]
    project2_response = client.post("/projects/", json={"name": "Project Two"})
    project2_id = project2_response.json()["id"]

    # Create tasks for each project
    client.post("/tasks/", json={"title": "Task 1 P1", "project_id": project1_id})
    client.post("/tasks/", json={"title": "Task 2 P1", "project_id": project1_id})
    client.post("/tasks/", json={"title": "Task 1 P2", "project_id": project2_id})

    response = client.get(f"/tasks/?project_id={project1_id}")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 2
    for task in tasks:
        assert task["project_id"] == project1_id

def test_read_tasks_by_status():
    # Create a project
    project_response = client.post("/projects/", json={"name": "Status Project"})
    project_id = project_response.json()["id"]

    # Create tasks with different statuses
    client.post("/tasks/", json={"title": "Pending Task", "status": "pending", "project_id": project_id})
    client.post("/tasks/", json={"title": "Completed Task", "status": "completed", "project_id": project_id})

    response = client.get(f"/tasks/?status=pending")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Pending Task"

def test_read_task():
    # First create a project and a task
    project_response = client.post(
        "/projects/",
        json={"name": "Project for Task Read"},
    )
    project_id = project_response.json()["id"]

    task_response = client.post(
        "/tasks/",
        json={
            "title": "Task to Read",
            "description": "Description for task to read.",
            "status": "pending",
            "project_id": project_id,
        },
    )
    task_id = task_response.json()["id"]

    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Task to Read"
    assert data["id"] == task_id

def test_read_nonexistent_task():
    response = client.get("/tasks/9999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}

def test_update_task():
    # First create a project and a task
    project_response = client.post(
        "/projects/",
        json={"name": "Project for Task Update"},
    )
    project_id = project_response.json()["id"]

    task_response = client.post(
        "/tasks/",
        json={
            "title": "Task to Update",
            "description": "Description for task to update.",
            "status": "pending",
            "project_id": project_id,
        },
    )
    task_id = task_response.json()["id"]

    response = client.put(
        f"/tasks/{task_id}",
        json={
            "title": "Updated Task Title",
            "status": "completed",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Task Title"
    assert data["status"] == "completed"
    assert data["id"] == task_id

def test_delete_task():
    # First create a project and a task
    project_response = client.post(
        "/projects/",
        json={"name": "Project for Task Delete"},
    )
    project_id = project_response.json()["id"]

    task_response = client.post(
        "/tasks/",
        json={
            "title": "Task to Delete",
            "description": "Description for task to delete.",
            "status": "pending",
            "project_id": project_id,
        },
    )
    task_id = task_response.json()["id"]

    response = client.delete(f"/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json() == {"message": "Task deleted successfully"}

    # Verify it's deleted
    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 404
