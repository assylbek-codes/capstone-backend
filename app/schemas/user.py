from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field

from app.schemas.base import BaseSchema, BaseCreateSchema, BaseUpdateSchema


class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False


class UserCreate(BaseCreateSchema):
    email: EmailStr
    username: str
    password: str


class UserUpdate(BaseUpdateSchema):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class UserInDBBase(UserBase, BaseSchema):
    class Config:
        from_attributes = True


class User(UserInDBBase):
    pass


class UserInDB(UserInDBBase):
    hashed_password: str 