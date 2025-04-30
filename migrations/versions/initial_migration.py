"""Initial migration with email verification

Revision ID: initial_migration
Revises: 
Create Date: 2024-05-05

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create users table with email verification fields
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('is_verified', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('verification_code', sa.String(), nullable=True),
        sa.Column('verification_code_expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create environments table
    op.create_table('environments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('dimensions', sa.JSON(), nullable=False),
        sa.Column('elements', sa.JSON(), nullable=False),
        sa.Column('graph', sa.JSON(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_environments_id'), 'environments', ['id'], unique=False)

    # Create scenarios table
    op.create_table('scenarios',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('environment_id', sa.Integer(), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['environment_id'], ['environments.id'], ),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_scenarios_id'), 'scenarios', ['id'], unique=False)

    # Create algos table
    op.create_table('algos',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('module_path', sa.String(), nullable=False),
        sa.Column('function_name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_algos_id'), 'algos', ['id'], unique=False)

    # Create tasks table
    op.create_table('tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('task_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=False),
        sa.Column('scenario_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tasks_id'), 'tasks', ['id'], unique=False)

    # Create solves table
    op.create_table('solves',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('scenario_id', sa.Integer(), nullable=False),
        sa.Column('environment_id', sa.Integer(), nullable=False),
        sa.Column('algorithm_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('results', sa.JSON(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.Column('timeout', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('celery_task_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['algorithm_id'], ['algos.id'], ),
        sa.ForeignKeyConstraint(['environment_id'], ['environments.id'], ),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['scenario_id'], ['scenarios.id'], ),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_solves_id'), 'solves', ['id'], unique=False)

    # Add initial algorithms data
    op.bulk_insert(
        sa.table('algos',
            sa.column('name', sa.String),
            sa.column('description', sa.Text),
            sa.column('is_active', sa.Boolean),
            sa.column('parameters', sa.JSON),
            sa.column('module_path', sa.String),
            sa.column('function_name', sa.String)
        ),
        [
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
        },
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
    )


def downgrade():
    # Drop tables in reverse order
    op.drop_table('solves')
    op.drop_table('tasks')
    op.drop_table('algos')
    op.drop_table('scenarios')
    op.drop_table('environments')
    op.drop_table('users') 