from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema


class AlgoBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = True
    parameters: Optional[Dict[str, Any]] = None
    module_path: Optional[str] = None
    function_name: Optional[str] = None


class AlgoCreate(BaseCreateSchema):
    name: str
    description: Optional[str] = None
    is_active: bool = True
    parameters: Dict[str, Any]
    module_path: str
    function_name: str


class AlgoUpdate(BaseUpdateSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None
    module_path: Optional[str] = None
    function_name: Optional[str] = None


class AlgoInDBBase(AlgoBase, BaseSchema):
    class Config:
        from_attributes = True


class Algo(AlgoInDBBase):
    pass


class AlgoWithDetails(Algo):
    solves_count: int = 0 