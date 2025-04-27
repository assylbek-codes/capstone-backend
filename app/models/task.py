from sqlalchemy import Column, Integer, String, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from app.db.base import Base, TimestampMixin


class TaskStatus(enum.Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    task_type = Column(String, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    
    # Task details - starting point, ending point, robot, priority, etc.
    details = Column(JSON, nullable=False)
    
    # Link to the scenario
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    
    # Relationships
    scenario = relationship("Scenario", back_populates="tasks")
    solves = relationship("Solve", back_populates="task") 