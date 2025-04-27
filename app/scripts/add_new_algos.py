#!/usr/bin/env python
"""
This script adds two new task assignment algorithms to the database.
"""
import sys
import os

# Add parent directory to path to import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy.orm import Session
from app.db.base import SessionLocal
from app.models.algo import Algo

def add_algorithms(db: Session):
    """Add new algorithms to the database if they don't exist."""
    
    algos = [
        {
            "name": "Makespan Balanced Assignment",
            "description": "Assigns each task to the robot with the current smallest total path length, minimizing the maximum completion time across all robots.",
            "is_active": True,
            "parameters": {
                "description": "Algorithm that balances the makespan (maximum completion time) across robots."
            },
            "module_path": "app.tasks.solve",
            "function_name": "makespan_balanced_assignment"
        },
        {
            "name": "Marginal Benefit Assignment",
            "description": "Assigns tasks to robots whose total travel distance increases the least by adding the task, optimizing for global efficiency.",
            "is_active": True,
            "parameters": {
                "description": "Algorithm that minimizes the marginal increase in travel distance for each task assignment."
            },
            "module_path": "app.tasks.solve",
            "function_name": "marginal_benefit_assignment"
        }
    ]
    
    for algo_data in algos:
        # Check if algo already exists
        existing = db.query(Algo).filter(Algo.name == algo_data["name"]).first()
        if existing:
            print(f"Algorithm '{algo_data['name']}' already exists.")
            continue
        
        # Create new algo
        algo = Algo(
            name=algo_data["name"],
            description=algo_data["description"],
            is_active=algo_data["is_active"],
            parameters=algo_data["parameters"],
            module_path=algo_data["module_path"],
            function_name=algo_data["function_name"]
        )
        
        db.add(algo)
        print(f"Added algorithm: {algo_data['name']}")
    
    db.commit()
    print("All algorithms added successfully.")

def main():
    """Main entry point for the script."""
    db = SessionLocal()
    try:
        add_algorithms(db)
    finally:
        db.close()

if __name__ == "__main__":
    main() 