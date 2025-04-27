from sqlalchemy import Column, Integer, String, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from app.db.base import Base, TimestampMixin


class SolveStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Solve(Base, TimestampMixin):
    __tablename__ = "solves"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(Enum(SolveStatus), nullable=False, default=SolveStatus.PENDING)
    
    # Parameters used for the solve
    parameters = Column(JSON, nullable=False)
    
    # Results of the solve (paths, metrics, etc.)
    results = Column(JSON, nullable=True)
    
    # Foreign keys
    environment_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    algorithm_id = Column(Integer, ForeignKey("algos.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    
    # Celery task ID
    celery_task_id = Column(String, nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="solves")
    environment = relationship("Environment", back_populates="solves")
    scenario = relationship("Scenario", back_populates="solves")
    algorithm = relationship("Algo", back_populates="solves")
    task = relationship("Task", back_populates="solves") 