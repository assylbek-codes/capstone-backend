from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema
from app.models.solve import SolveStatus
from app.schemas.task import Task


class SolveBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[SolveStatus] = None
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    environment_id: Optional[int] = None
    scenario_id: Optional[int] = None
    algorithm_id: Optional[int] = None
    task_id: Optional[int] = None
    celery_task_id: Optional[str] = None


class SolveCreate(BaseCreateSchema):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]
    environment_id: int
    scenario_id: int
    algorithm_id: int
    task_id: int


class SolveUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[SolveStatus] = None
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    celery_task_id: Optional[str] = None


class SolveInDBBase(SolveBase, BaseSchema):
    owner_id: int

    class Config:
        from_attributes = True


class Solve(SolveInDBBase):
    pass


class SolveWithDetails(Solve):
    task: Optional[Task] = None 