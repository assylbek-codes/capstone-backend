from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TimestampSchema(BaseModel):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BaseSchema(TimestampSchema):
    id: Optional[int] = None


class BaseCreateSchema(BaseModel):
    pass


class BaseUpdateSchema(BaseModel):
    pass 