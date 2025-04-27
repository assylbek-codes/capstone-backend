#!/usr/bin/env python
"""
This script adds the predefined task assignment algorithms to the database.
"""
import sys
import os

# Add parent directory to path to import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy.orm import Session
from app.db.base import SessionLocal
from app.models.algo import Algo
from app.models.user import User

def add_algorithms(db: Session):
    """Add predefined algorithms to the database if they don't exist."""
    
    algos = [
        {
            "name": "Greedy Distance Minimization",
            "description": "A simple greedy algorithm that assigns tasks to the closest available robot, minimizing the total travel distance for each assignment.",
            "is_active": True,
            "parameters": {
                "description": "Simple algorithm with no additional parameters required."
            },
            "module_path": "app.tasks.solve",
            "function_name": "greedy_assignment"
        },
        {
            "name": "Global Optimization with Hungarian Algorithm",
            "description": "Uses the Hungarian algorithm to find the globally optimal assignment of tasks to robots, minimizing the total cost across all assignments.",
            "is_active": True,
            "parameters": {
                "description": "Globally optimal algorithm with no additional parameters required."
            },
            "module_path": "app.tasks.solve",
            "function_name": "hungarian_assignment"
        },
        {
            "name": "K-Means Inspired Task Clustering",
            "description": "Groups tasks into clusters based on spatial proximity using K-means, then assigns each cluster to a robot.",
            "is_active": True,
            "parameters": {
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters to form. If None, uses the number of robots.",
                    "default": None,
                    "required": False
                }
            },
            "module_path": "app.tasks.solve",
            "function_name": "kmeans_clustering_assignment"
        },
        {
            "name": "Auction-Based Allocation",
            "description": "Simulates an auction where robots bid for tasks based on their cost to complete them.",
            "is_active": True,
            "parameters": {
                "description": "Distributed auction-based algorithm with no additional parameters required."
            },
            "module_path": "app.tasks.solve",
            "function_name": "auction_based_assignment"
        },
        {
            "name": "Genetic Algorithm for Assignment",
            "description": "Uses a genetic algorithm to evolve an optimal assignment of tasks to robots over multiple generations.",
            "is_active": True,
            "parameters": {
                "population_size": {
                    "type": "integer",
                    "description": "Size of the population in the genetic algorithm",
                    "default": 50,
                    "required": False
                },
                "generations": {
                    "type": "integer",
                    "description": "Number of generations to evolve",
                    "default": 100,
                    "required": False
                },
                "mutation_rate": {
                    "type": "number",
                    "description": "Probability of mutation for each gene",
                    "default": 0.1,
                    "required": False
                }
            },
            "module_path": "app.tasks.solve",
            "function_name": "genetic_algorithm_assignment"
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