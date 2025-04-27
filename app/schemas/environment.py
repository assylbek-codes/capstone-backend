from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema


class EnvironmentBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    dimensions: Optional[Dict[str, Any]] = None
    elements: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


class EnvironmentCreate(BaseCreateSchema):
    name: str
    description: Optional[str] = None
    dimensions: Dict[str, Any]
    elements: Dict[str, Any]
    graph: Dict[str, Any]


class EnvironmentUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    dimensions: Optional[Dict[str, Any]] = None
    elements: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None


class EnvironmentInDBBase(EnvironmentBase, BaseSchema):
    owner_id: int

    class Config:
        from_attributes = True


class Environment(EnvironmentInDBBase):
    pass


class EnvironmentWithDetails(Environment):
    scenarios_count: int = 0
    solves_count: int = 0 