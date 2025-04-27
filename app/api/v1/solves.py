from typing import Any, List, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.db.base import get_db
from app.models.solve import Solve, SolveStatus
from app.models.environment import Environment
from app.models.scenario import Scenario
from app.models.task import Task
from app.models.algo import Algo
from app.models.user import User
from app.schemas.solve import Solve as SolveSchema
from app.schemas.solve import SolveCreate, SolveUpdate, SolveWithDetails
from app.core.deps import get_current_active_user
from app.tasks.solve import execute_solve

router = APIRouter()


# Create CRUD object
solve_crud = CRUDBase[Solve, SolveCreate, SolveUpdate](Solve)


@router.get("/", response_model=List[SolveSchema])
def read_solves(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    environment_id: int = Query(None, description="Filter by environment ID"),
    scenario_id: int = Query(None, description="Filter by scenario ID"),
) -> Any:
    """
    Retrieve solves.
    """
    query = db.query(Solve).filter(Solve.owner_id == current_user.id)
    
    if environment_id:
        query = query.filter(Solve.environment_id == environment_id)
    
    if scenario_id:
        query = query.filter(Solve.scenario_id == scenario_id)
    
    solves = query.offset(skip).limit(limit).all()
    return solves


@router.post("/", response_model=SolveSchema)
def create_solve(
    *,
    db: Session = Depends(get_db),
    solve_in: SolveCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create new solve.
    """
    # Verify environment ownership
    environment = db.query(Environment).filter(
        Environment.id == solve_in.environment_id,
        Environment.owner_id == current_user.id
    ).first()
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found or not owned by user")
    
    # Verify scenario ownership
    scenario = db.query(Scenario).filter(
        Scenario.id == solve_in.scenario_id,
        Scenario.owner_id == current_user.id
    ).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found or not owned by user")
    
    # Verify algorithm exists and is active
    algorithm = db.query(Algo).filter(
        Algo.id == solve_in.algorithm_id,
        Algo.is_active == True
    ).first()
    if not algorithm:
        raise HTTPException(status_code=404, detail="Algorithm not found or not active")
    
    # Verify task exists and belongs to the specified scenario
    task = db.query(Task).filter(
        Task.id == solve_in.task_id,
        Task.scenario_id == solve_in.scenario_id
    ).first()
    
    if not task:
        raise HTTPException(
            status_code=400, 
            detail="Task not found or doesn't belong to the specified scenario"
        )
    
    # Create the solve record
    solve_obj = Solve(
        name=solve_in.name,
        description=solve_in.description,
        status=SolveStatus.PENDING,
        parameters=solve_in.parameters,
        environment_id=solve_in.environment_id,
        scenario_id=solve_in.scenario_id,
        algorithm_id=solve_in.algorithm_id,
        task_id=solve_in.task_id,
        owner_id=current_user.id,
    )
    
    db.add(solve_obj)
    db.commit()
    db.refresh(solve_obj)
    
    # Submit the solve task to Celery
    print(f"Submitting solve task for solve_id={solve_obj.id}")
    
    # For debugging, let's try executing the task directly
    try:
        # This will run the task synchronously for debugging
        direct_result = execute_solve(solve_obj.id)
        print(f"Direct task execution result: {direct_result}")
    except Exception as e:
        print(f"Error in direct execution: {str(e)}")
    
    # Now submit it to Celery as usual
    task_result = execute_solve.delay(solve_obj.id)
    print(f"Submitted task with task_id={task_result.id}")
    
    # Update the solve record with the Celery task ID
    solve_obj.celery_task_id = task_result.id
    db.add(solve_obj)
    db.commit()
    db.refresh(solve_obj)
    
    return solve_obj


@router.get("/{id}", response_model=SolveWithDetails)
def read_solve(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get solve by ID.
    """
    solve = solve_crud.get(db=db, id=id)
    if not solve:
        raise HTTPException(status_code=404, detail="Solve not found")
    if solve.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Load related tasks
    return solve


@router.put("/{id}", response_model=SolveSchema)
def update_solve(
    *,
    db: Session = Depends(get_db),
    id: int,
    solve_in: SolveUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update a solve (limited to name and description).
    """
    solve = solve_crud.get(db=db, id=id)
    if not solve:
        raise HTTPException(status_code=404, detail="Solve not found")
    if solve.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Only allow updating name and description
    # Create a new dict with only the allowed fields
    update_data = {}
    if solve_in.name is not None:
        update_data["name"] = solve_in.name
    if solve_in.description is not None:
        update_data["description"] = solve_in.description
    
    solve = solve_crud.update(db=db, db_obj=solve, obj_in=update_data)
    return solve


@router.delete("/{id}", response_model=SolveSchema)
def delete_solve(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete a solve.
    """
    solve = solve_crud.get(db=db, id=id)
    if not solve:
        raise HTTPException(status_code=404, detail="Solve not found")
    if solve.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Can't delete a solve in progress
    if solve.status == SolveStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a solve that is currently running"
        )
    
    solve = solve_crud.remove(db=db, id=id)
    return solve 