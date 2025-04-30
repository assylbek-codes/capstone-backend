from sqlalchemy import Boolean, Column, String, Integer, DateTime
from sqlalchemy.orm import relationship
import datetime

from app.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Email verification fields
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)
    verification_code_expires_at = Column(DateTime, nullable=True)

    # Relationships
    environments = relationship("Environment", back_populates="owner", cascade="all, delete-orphan")
    scenarios = relationship("Scenario", back_populates="owner", cascade="all, delete-orphan")
    solves = relationship("Solve", back_populates="owner", cascade="all, delete-orphan") 