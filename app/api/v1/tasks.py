from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.db.base import get_db
from app.models.task import Task
from app.models.scenario import Scenario
from app.models.user import User
from app.schemas.task import Task as TaskSchema, TaskCreate, TaskUpdate, TaskBatchCreate
from app.core.deps import get_current_active_user
from app.services.llm import generate_tasks

router = APIRouter()


# Create CRUD object
task_crud = CRUDBase[Task, TaskCreate, TaskUpdate](Task)


@router.get("/", response_model=List[TaskSchema])
def read_tasks(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    scenario_id: int = Query(None, description="Filter by scenario ID"),
) -> Any:
    """
    Retrieve tasks.
    """
    if scenario_id:
        # Check if scenario exists and belongs to current user
        scenario = db.query(Scenario).filter(
            Scenario.id == scenario_id,
            Scenario.owner_id == current_user.id
        ).first()
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        # Get tasks for this scenario
        tasks = db.query(Task).filter(
            Task.scenario_id == scenario_id
        ).offset(skip).limit(limit).all()
    else:
        # Get all tasks for current user's scenarios
        tasks = db.query(Task).join(Scenario).filter(
            Scenario.owner_id == current_user.id
        ).offset(skip).limit(limit).all()
    
    return tasks


@router.post("/", response_model=TaskSchema)
def create_task(
    *,
    db: Session = Depends(get_db),
    task_in: TaskCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create new task.
    """
    # Check if scenario exists and belongs to current user
    scenario = db.query(Scenario).filter(
        Scenario.id == task_in.scenario_id,
        Scenario.owner_id == current_user.id
    ).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found or not owned by user")
    
    task = task_crud.create(db=db, obj_in=task_in)
    return task


@router.post("/batch", response_model=List[TaskSchema])
def create_tasks_batch(
    *,
    db: Session = Depends(get_db),
    tasks_in: TaskBatchCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create multiple tasks at once.
    """
    created_tasks = []
    
    # Check the first task's scenario to verify ownership
    if tasks_in.tasks and len(tasks_in.tasks) > 0:
        first_task = tasks_in.tasks[0]
        scenario = db.query(Scenario).filter(
            Scenario.id == first_task.scenario_id,
            Scenario.owner_id == current_user.id
        ).first()
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found or not owned by user")
        
        # Verify all tasks have the same scenario_id
        scenario_ids = {task.scenario_id for task in tasks_in.tasks}
        if len(scenario_ids) > 1:
            raise HTTPException(
                status_code=400, 
                detail="All tasks in a batch must belong to the same scenario"
            )
        
        # Create all tasks
        for task_in in tasks_in.tasks:
            task = task_crud.create(db=db, obj_in=task_in)
            created_tasks.append(task)
    
    return created_tasks


@router.post("/generate", response_model=List[TaskSchema])
async def generate_tasks_with_llm(
    *,
    db: Session = Depends(get_db),
    scenario_id: int,
    num_tasks: int = Query(5, gt=0, le=50),
    task_type: Optional[str] = Query(None, description="Optional task type filter"),
    name: Optional[str] = Query(None, description="Optional name for the tasks"),
    description: Optional[str] = Query(None, description="Optional description for the tasks"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Generate tasks using LLM based on environment and scenario.
    """
    print(f"Generating {num_tasks} tasks for scenario {scenario_id}")
    # Check if scenario exists and belongs to current user
    scenario = db.query(Scenario).filter(
        Scenario.id == scenario_id,
        Scenario.owner_id == current_user.id
    ).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found or not owned by user")
    
    # Get the environment data
    environment = scenario.environment
    
    # Verify task_type is valid if provided
    valid_task_types = ["pickup_delivery"]
    if task_type and task_type not in valid_task_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task type. Must be one of: {', '.join(valid_task_types)}"
        )
    
    # Use LLM to generate tasks
    generated_task_data = await generate_tasks(
        environment_data={
            "dimensions": environment.dimensions,
            "elements": environment.elements,
            "graph": environment.graph
        },
        scenario_data={
            "name": scenario.name,
            "description": scenario.description,
            # "parameters": scenario.parameters
        },
        num_tasks=num_tasks,
        task_type=task_type
    )
    
    # Convert generated tasks to DB models and save
    created_tasks = []

    task_in = TaskCreate(
        name=name,
        description=description,
        task_type=task_type,
        details=generated_task_data,
        scenario_id=scenario_id
    )
    task = task_crud.create(db=db, obj_in=task_in)
    created_tasks.append(task)
    
    return created_tasks


@router.get("/{id}", response_model=TaskSchema)
def read_task(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get task by ID.
    """
    task = task_crud.get(db=db, id=id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if user owns the scenario that contains this task
    scenario = db.query(Scenario).filter(
        Scenario.id == task.scenario_id,
        Scenario.owner_id == current_user.id
    ).first()
    if not scenario:
        raise HTTPException(status_code=403, detail="Not enough permissions")
        
    return task


@router.put("/{id}", response_model=TaskSchema)
def update_task(
    *,
    db: Session = Depends(get_db),
    id: int,
    task_in: TaskUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update a task.
    """
    task = task_crud.get(db=db, id=id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if user owns the scenario that contains this task
    scenario = db.query(Scenario).filter(
        Scenario.id == task.scenario_id,
        Scenario.owner_id == current_user.id
    ).first()
    if not scenario:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    task = task_crud.update(db=db, db_obj=task, obj_in=task_in)
    return task


@router.delete("/{id}", response_model=TaskSchema)
def delete_task(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete a task.
    """
    task = task_crud.get(db=db, id=id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if user owns the scenario that contains this task
    scenario = db.query(Scenario).filter(
        Scenario.id == task.scenario_id,
        Scenario.owner_id == current_user.id
    ).first()
    if not scenario:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    task = task_crud.remove(db=db, id=id)
    return task 