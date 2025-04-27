"""Change task_type to string

Revision ID: change_task_type_to_string
Revises: 8efd47645c55
Create Date: 2025-04-21 23:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'change_task_type_to_string'
down_revision: Union[str, None] = '8efd47645c55'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to change task_type from enum to string."""
    # Create a temporary column as string
    op.add_column('tasks', sa.Column('task_type_str', sa.String(), nullable=True))
    
    # Copy data from enum to string
    op.execute("""
    UPDATE tasks
    SET task_type_str = task_type::text
    """)
    
    # Set not nullable constraint
    op.alter_column('tasks', 'task_type_str', nullable=False)
    
    # Drop the enum column
    op.drop_column('tasks', 'task_type')
    
    # Rename string column to original name
    op.alter_column('tasks', 'task_type_str', new_column_name='task_type')


def downgrade() -> None:
    """Downgrade schema to revert task_type from string to enum."""
    # Create enum type if it doesn't exist
    op.execute("CREATE TYPE IF NOT EXISTS tasktype AS ENUM ('pickup', 'delivery', 'pickup_delivery')")
    
    # Create a temporary column as enum
    op.add_column('tasks', sa.Column('task_type_enum', sa.Enum('pickup', 'delivery', 'pickup_delivery', name='tasktype'), nullable=True))
    
    # Copy data from string to enum
    op.execute("""
    UPDATE tasks
    SET task_type_enum = task_type::tasktype
    """)
    
    # Set not nullable constraint
    op.alter_column('tasks', 'task_type_enum', nullable=False)
    
    # Drop the string column
    op.drop_column('tasks', 'task_type')
    
    # Rename enum column to original name
    op.alter_column('tasks', 'task_type_enum', new_column_name='task_type') 