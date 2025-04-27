from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum

from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema
from app.models.task import TaskStatus


class TaskBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    status: Optional[TaskStatus] = None
    details: Optional[Dict[str, Any]] = None
    scenario_id: Optional[int] = None


class TaskCreate(BaseCreateSchema):
    name: str
    description: Optional[str] = None
    task_type: str
    details: Dict[str, Any]
    scenario_id: int


class TaskBatchCreate(BaseModel):
    tasks: List[TaskCreate]


class TaskUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    status: Optional[TaskStatus] = None
    details: Optional[Dict[str, Any]] = None


class TaskInDBBase(TaskBase, BaseSchema):
    class Config:
        from_attributes = True


class Task(TaskInDBBase):
    pass


class TaskWithDetails(Task):
    solves_count: int = 0 