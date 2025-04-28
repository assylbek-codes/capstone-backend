from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.db.base import get_db
from app.models.algo import Algo
from app.models.user import User
from app.schemas.algo import Algo as AlgoSchema
from app.schemas.algo import AlgoCreate, AlgoUpdate
from app.core.deps import get_current_active_user, get_current_active_superuser

router = APIRouter()


# Create CRUD object
algo_crud = CRUDBase[Algo, AlgoCreate, AlgoUpdate](Algo)


@router.get("", response_model=List[AlgoSchema])
def read_algos(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Retrieve algorithms.
    """
    # Return only active algorithms
    algos = db.query(Algo).filter(Algo.is_active == True).offset(skip).limit(limit).all()
    return algos


@router.post("", response_model=AlgoSchema)
def create_algo(
    *,
    db: Session = Depends(get_db),
    algo_in: AlgoCreate,
    current_user: User = Depends(get_current_active_superuser),
) -> Any:
    """
    Create new algorithm (superuser only).
    """
    # Check if algorithm with this name already exists
    existing_algo = db.query(Algo).filter(Algo.name == algo_in.name).first()
    if existing_algo:
        raise HTTPException(
            status_code=400,
            detail="Algorithm with this name already exists"
        )
    
    algo = algo_crud.create(db=db, obj_in=algo_in)
    return algo


@router.get("/{id}", response_model=AlgoSchema)
def read_algo(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get algorithm by ID.
    """
    algo = algo_crud.get(db=db, id=id)
    if not algo:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    if not algo.is_active:
        # Regular users can only see active algorithms
        if not current_user.is_superuser:
            raise HTTPException(status_code=404, detail="Algorithm not found")
    return algo


@router.put("/{id}", response_model=AlgoSchema)
def update_algo(
    *,
    db: Session = Depends(get_db),
    id: int,
    algo_in: AlgoUpdate,
    current_user: User = Depends(get_current_active_superuser),
) -> Any:
    """
    Update an algorithm (superuser only).
    """
    algo = algo_crud.get(db=db, id=id)
    if not algo:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    # If name is being changed, check for conflicts
    if algo_in.name is not None and algo_in.name != algo.name:
        existing_algo = db.query(Algo).filter(Algo.name == algo_in.name).first()
        if existing_algo:
            raise HTTPException(
                status_code=400,
                detail="Algorithm with this name already exists"
            )
    
    algo = algo_crud.update(db=db, db_obj=algo, obj_in=algo_in)
    return algo


@router.delete("/{id}", response_model=AlgoSchema)
def delete_algo(
    *,
    db: Session = Depends(get_db),
    id: int,
    current_user: User = Depends(get_current_active_superuser),
) -> Any:
    """
    Delete an algorithm (superuser only).
    """
    algo = algo_crud.get(db=db, id=id)
    if not algo:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    # Instead of actually deleting, just mark as inactive
    algo.is_active = False
    db.add(algo)
    db.commit()
    db.refresh(algo)
    
    return algo 
