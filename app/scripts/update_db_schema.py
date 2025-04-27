#!/usr/bin/env python
"""
This script updates the database schema to change the relationship between Solve and Task
from many-to-many to one-to-many.
"""
import sys
import os

# Add parent directory to path to import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import text
from app.db.base import SessionLocal

def update_schema():
    """
    Updates the database schema:
    1. Adds a task_id column to the solves table
    2. Migrates data from solve_tasks to solves.task_id
    3. Drops the solve_tasks table
    """
    db = SessionLocal()
    try:
        # Check if task_id column already exists
        check_column = text("SELECT column_name FROM information_schema.columns WHERE table_name='solves' AND column_name='task_id'")
        result = db.execute(check_column)
        if not result.fetchone():
            print("Adding task_id column to solves table...")
            # Add task_id column to solves table
            add_column = text("ALTER TABLE solves ADD COLUMN task_id INTEGER REFERENCES tasks(id)")
            db.execute(add_column)
            
            # Migrate data from solve_tasks to solves.task_id
            print("Migrating data from solve_tasks to solves.task_id...")
            migrate_data = text("""
                UPDATE solves s
                SET task_id = st.task_id
                FROM solve_tasks st
                WHERE s.id = st.solve_id
            """)
            db.execute(migrate_data)
            
            # Make task_id not nullable
            not_null = text("ALTER TABLE solves ALTER COLUMN task_id SET NOT NULL")
            db.execute(not_null)
            
            # Drop the solve_tasks table
            print("Dropping solve_tasks table...")
            drop_table = text("DROP TABLE IF EXISTS solve_tasks")
            db.execute(drop_table)
            
            db.commit()
            print("Database schema updated successfully!")
        else:
            print("task_id column already exists in solves table. Schema is up to date.")
    except Exception as e:
        db.rollback()
        print(f"Error updating schema: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Updating database schema...")
    update_schema()
    print("Done!") 