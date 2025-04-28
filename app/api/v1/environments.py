from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.db.base import get_db
from app.models.environment import Environment
from app.models.user import User
from app.schemas.environment import Environment as EnvironmentSchema
from app.schemas.environment import EnvironmentCreate, EnvironmentUpdate
from app.core.deps import get_current_active_user

router = APIRouter()


# Create CRUD object
environment_crud = CRUDBase[Environment, EnvironmentCreate, EnvironmentUpdate](Environment)


@router.get("", response_model=List[EnvironmentSchema])
def read_environments(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Retrieve environments.
    """
    environments = environment_crud.get_multi_by_owner(
        db=db, owner_id=current_user.id, skip=skip, limit=limit
    )
    return environments


@router.post("", response_model=EnvironmentSchema)
def create_environment(
    *,
    db: Session = Depends(get_db),
    environment_in: EnvironmentCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create new environment.
    """
    environment = environment_crud.create(db=db, obj_in=environment_in, owner_id=current_user.id)
    return environment


@router.get("/{id}", response_model=EnvironmentSchema)
def read_environment(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get environment by ID.
    """
    environment = environment_crud.get(db=db, id=id)
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found")
    if environment.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return environment


@router.put("/{id}", response_model=EnvironmentSchema)
def update_environment(
    *,
    db: Session = Depends(get_db),
    id: int,
    environment_in: EnvironmentUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update an environment.
    """
    environment = environment_crud.get(db=db, id=id)
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found")
    if environment.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    environment = environment_crud.update(db=db, db_obj=environment, obj_in=environment_in)
    return environment


@router.delete("/{id}", response_model=EnvironmentSchema)
def delete_environment(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete an environment.
    """
    environment = environment_crud.get(db=db, id=id)
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found")
    if environment.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    environment = environment_crud.remove(db=db, id=id)
    return environment 
