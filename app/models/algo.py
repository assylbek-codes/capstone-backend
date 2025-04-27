from sqlalchemy import Column, Integer, String, Text, JSON, Boolean
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin


class Algo(Base, TimestampMixin):
    __tablename__ = "algos"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Algorithm parameters and configuration
    parameters = Column(JSON, nullable=False)
    
    # Entry point in code (path to module, function name)
    module_path = Column(String, nullable=False)
    function_name = Column(String, nullable=False)
    
    # Relationships
    solves = relationship("Solve", back_populates="algorithm") 