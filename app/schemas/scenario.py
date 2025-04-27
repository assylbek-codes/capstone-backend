from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema


class ScenarioBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    environment_id: Optional[int] = None


class ScenarioCreate(BaseCreateSchema):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]
    environment_id: int


class ScenarioUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    environment_id: Optional[int] = None


class ScenarioInDBBase(ScenarioBase, BaseSchema):
    owner_id: int

    class Config:
        from_attributes = True


class Scenario(ScenarioInDBBase):
    pass


class ScenarioWithDetails(Scenario):
    tasks_count: int = 0
    solves_count: int = 0 