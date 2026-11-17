from pydantic import BaseModel
from typing import List, Optional

class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "pending"

class TaskCreate(TaskBase):
    project_id: int

class Task(TaskBase):
    id: int
    project_id: int

    class Config:
        from_attributes = True

class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    tasks: List[Task] = []

    class Config:
        from_attributes = True
