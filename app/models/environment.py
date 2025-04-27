from sqlalchemy import Column, Integer, String, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin


class Environment(Base, TimestampMixin):
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    dimensions = Column(JSON, nullable=False)  # Store width, height
    elements = Column(JSON, nullable=False)  # Store shelves, drop-offs, robot stations, etc.
    graph = Column(JSON, nullable=False)  # Store graph representation for algorithms
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    owner = relationship("User", back_populates="environments")
    scenarios = relationship("Scenario", back_populates="environment", cascade="all, delete-orphan")
    solves = relationship("Solve", back_populates="environment", cascade="all, delete-orphan") 