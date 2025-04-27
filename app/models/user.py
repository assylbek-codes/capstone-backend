from sqlalchemy import Boolean, Column, String, Integer
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    # Relationships
    environments = relationship("Environment", back_populates="owner", cascade="all, delete-orphan")
    scenarios = relationship("Scenario", back_populates="owner", cascade="all, delete-orphan")
    solves = relationship("Solve", back_populates="owner", cascade="all, delete-orphan") 