"""integrate_script_changes

Revision ID: 01e30a91af55
Revises: change_task_type_to_string
Create Date: 2025-04-29 12:16:38.171411

"""
from typing import Sequence, Union
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from sqlalchemy import String, Integer, Boolean, JSON, Text


# revision identifiers, used by Alembic.
revision: str = '01e30a91af55'
down_revision: Union[str, None] = 'change_task_type_to_string'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema and add initial data."""
    
    # 1. Update schema: Change Solve-Task relation from M:M to 1:M
    # Add task_id column to solves table
    op.add_column('solves', sa.Column('task_id', sa.Integer(), nullable=True))
    
    # Add foreign key constraint
    op.create_foreign_key(
        "fk_solves_task_id_tasks", 
        "solves", "tasks", 
        ["task_id"], ["id"]
    )
    
    # Migrate data from solve_tasks to solves.task_id
    op.execute("""
        UPDATE solves s
        SET task_id = st.task_id
        FROM solve_tasks st
        WHERE s.id = st.solve_id
    """)
    
    # Make task_id not nullable
    op.alter_column('solves', 'task_id', nullable=False)
    
    # Drop the solve_tasks table
    op.drop_table('solve_tasks')
    
    # 2. Add initial algorithms data
    # Define the algos table structure for bulk insert
    algos_table = table('algos',
        column('name', String),
        column('description', Text),
        column('is_active', Boolean),
        column('parameters', JSON),
        column('module_path', String),
        column('function_name', String)
    )
    
    # Algorithm data from both algorithm scripts
    algo_data = [
        # From add_algorithms.py
        {
            "name": "Greedy Distance Minimization",
            "description": "A simple greedy algorithm that assigns tasks to the closest available robot, minimizing the total travel distance for each assignment.",
            "is_active": True,
            "parameters": json.dumps({
                "description": "Simple algorithm with no additional parameters required."
            }),
            "module_path": "app.tasks.solve",
            "function_name": "greedy_assignment"
        },
        {
            "name": "Global Optimization with Hungarian Algorithm",
            "description": "Uses the Hungarian algorithm to find the globally optimal assignment of tasks to robots, minimizing the total cost across all assignments.",
            "is_active": True,
            "parameters": json.dumps({
                "description": "Globally optimal algorithm with no additional parameters required."
            }),
            "module_path": "app.tasks.solve",
            "function_name": "hungarian_assignment"
        },
        {
            "name": "K-Means Inspired Task Clustering",
            "description": "Groups tasks into clusters based on spatial proximity using K-means, then assigns each cluster to a robot.",
            "is_active": True,
            "parameters": json.dumps({
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters to form. If None, uses the number of robots.",
                    "default": None,
                    "required": False
                }
            }),
            "module_path": "app.tasks.solve",
            "function_name": "kmeans_clustering_assignment"
        },
        {
            "name": "Auction-Based Allocation",
            "description": "Simulates an auction where robots bid for tasks based on their cost to complete them.",
            "is_active": True,
            "parameters": json.dumps({
                "description": "Distributed auction-based algorithm with no additional parameters required."
            }),
            "module_path": "app.tasks.solve",
            "function_name": "auction_based_assignment"
        },
        {
            "name": "Genetic Algorithm for Assignment",
            "description": "Uses a genetic algorithm to evolve an optimal assignment of tasks to robots over multiple generations.",
            "is_active": True,
            "parameters": json.dumps({
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
            }),
            "module_path": "app.tasks.solve",
            "function_name": "genetic_algorithm_assignment"
        },
        
        # From add_new_algos.py
        {
            "name": "Makespan Balanced Assignment",
            "description": "Assigns each task to the robot with the current smallest total path length, minimizing the maximum completion time across all robots.",
            "is_active": True,
            "parameters": json.dumps({
                "description": "Algorithm that balances the makespan (maximum completion time) across robots."
            }),
            "module_path": "app.tasks.solve",
            "function_name": "makespan_balanced_assignment"
        },
        {
            "name": "Marginal Benefit Assignment",
            "description": "Assigns tasks to robots whose total travel distance increases the least by adding the task, optimizing for global efficiency.",
            "is_active": True,
            "parameters": json.dumps({
                "description": "Algorithm that minimizes the marginal increase in travel distance for each task assignment."
            }),
            "module_path": "app.tasks.solve",
            "function_name": "marginal_benefit_assignment"
        }
    ]
    
    # Insert all algorithms
    op.bulk_insert(algos_table, algo_data)


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Recreate the solve_tasks junction table
    op.create_table('solve_tasks',
        sa.Column('solve_id', sa.Integer(), nullable=False),
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['solve_id'], ['solves.id'], ),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('solve_id', 'task_id')
    )
    
    # 2. Migrate data back from solves.task_id to solve_tasks
    op.execute("""
        INSERT INTO solve_tasks (solve_id, task_id)
        SELECT id, task_id FROM solves
        WHERE task_id IS NOT NULL
    """)
    
    # 3. Drop the task_id column from solves
    op.drop_constraint('fk_solves_task_id_tasks', 'solves', type_='foreignkey')
    op.drop_column('solves', 'task_id')
    
    # 4. Delete the algorithm data that was added
    algo_names = [
        "Greedy Distance Minimization",
        "Global Optimization with Hungarian Algorithm",
        "K-Means Inspired Task Clustering",
        "Auction-Based Allocation", 
        "Genetic Algorithm for Assignment",
        "Makespan Balanced Assignment",
        "Marginal Benefit Assignment"
    ]
    
    for name in algo_names:
        op.execute(f"DELETE FROM algos WHERE name = '{name}'")
