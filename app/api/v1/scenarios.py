from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.db.base import get_db
from app.models.scenario import Scenario
from app.models.environment import Environment
from app.models.user import User
from app.schemas.scenario import Scenario as ScenarioSchema
from app.schemas.scenario import ScenarioCreate, ScenarioUpdate
from app.core.deps import get_current_active_user

router = APIRouter()


# Create CRUD object
scenario_crud = CRUDBase[Scenario, ScenarioCreate, ScenarioUpdate](Scenario)


@router.get("/", response_model=List[ScenarioSchema])
def read_scenarios(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    environment_id: int = Query(None, description="Filter by environment ID"),
) -> Any:
    """
    Retrieve scenarios.
    """
    if environment_id:
        # Check if environment exists and belongs to current user
        environment = db.query(Environment).filter(
            Environment.id == environment_id,
            Environment.owner_id == current_user.id
        ).first()
        if not environment:
            raise HTTPException(status_code=404, detail="Environment not found")
        
        # Get scenarios for this environment
        scenarios = db.query(Scenario).filter(
            Scenario.environment_id == environment_id,
            Scenario.owner_id == current_user.id
        ).offset(skip).limit(limit).all()
    else:
        # Get all scenarios for current user
        scenarios = scenario_crud.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    
    return scenarios


@router.post("/", response_model=ScenarioSchema)
def create_scenario(
    *,
    db: Session = Depends(get_db),
    scenario_in: ScenarioCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create new scenario.
    """
    # Check if environment exists and belongs to current user
    environment = db.query(Environment).filter(
        Environment.id == scenario_in.environment_id,
        Environment.owner_id == current_user.id
    ).first()
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found or not owned by user")
    
    scenario = scenario_crud.create(db=db, obj_in=scenario_in, owner_id=current_user.id)
    return scenario


@router.get("/{id}", response_model=ScenarioSchema)
def read_scenario(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get scenario by ID.
    """
    scenario = scenario_crud.get(db=db, id=id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if scenario.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return scenario


@router.put("/{id}", response_model=ScenarioSchema)
def update_scenario(
    *,
    db: Session = Depends(get_db),
    id: int,
    scenario_in: ScenarioUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update a scenario.
    """
    scenario = scenario_crud.get(db=db, id=id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if scenario.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # If environment_id is being updated, verify that the new environment belongs to user
    if scenario_in.environment_id is not None and scenario_in.environment_id != scenario.environment_id:
        environment = db.query(Environment).filter(
            Environment.id == scenario_in.environment_id,
            Environment.owner_id == current_user.id
        ).first()
        if not environment:
            raise HTTPException(
                status_code=404, 
                detail="New environment not found or not owned by user"
            )
    
    scenario = scenario_crud.update(db=db, db_obj=scenario, obj_in=scenario_in)
    return scenario


@router.delete("/{id}", response_model=ScenarioSchema)
def delete_scenario(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete a scenario.
    """
    scenario = scenario_crud.get(db=db, id=id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if scenario.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    scenario = scenario_crud.remove(db=db, id=id)
    return scenario 