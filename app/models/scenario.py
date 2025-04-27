from sqlalchemy import Column, Integer, String, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin


class Scenario(Base, TimestampMixin):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=False)  # Store scenario parameters
    environment_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    owner = relationship("User", back_populates="scenarios")
    environment = relationship("Environment", back_populates="scenarios")
    tasks = relationship("Task", back_populates="scenario", cascade="all, delete-orphan")
    solves = relationship("Solve", back_populates="scenario", cascade="all, delete-orphan") 