from typing import Any, Dict, Optional, Union
import random
import string
import datetime
from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
from app.crud.base import CRUDBase
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()
    
    def get_by_username(self, db: Session, *, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        verification_code = self.generate_verification_code()
        verification_expiry = datetime.datetime.now() + datetime.timedelta(minutes=30)
        
        db_obj = User(
            email=obj_in.email,
            username=obj_in.username,
            hashed_password=get_password_hash(obj_in.password),
            is_superuser=False,
            is_verified=False,
            verification_code=verification_code,
            verification_code_expires_at=verification_expiry
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: User, obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if update_data.get("password"):
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, email: str, password: str) -> Optional[User]:
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_active(self, user: User) -> bool:
        return user.is_active

    def is_superuser(self, user: User) -> bool:
        return user.is_superuser
        
    def generate_verification_code(self) -> str:
        """Generate a random 6-digit verification code"""
        return ''.join(random.choices(string.digits, k=6))
        
    def set_verification_code(self, db: Session, *, user_id: int) -> str:
        """Set a new verification code for a user"""
        user = self.get(db, id=user_id)
        if not user:
            return None
            
        verification_code = self.generate_verification_code()
        verification_expiry = datetime.datetime.now() + datetime.timedelta(minutes=30)
        
        user.verification_code = verification_code
        user.verification_code_expires_at = verification_expiry
        db.commit()
        db.refresh(user)
        
        return verification_code
        
    def verify_email(self, db: Session, *, email: str, code: str) -> bool:
        """Verify a user's email with the provided code"""
        user = self.get_by_email(db, email=email)
        if not user:
            return False
            
        # Check if code matches and hasn't expired
        if (user.verification_code == code and 
            user.verification_code_expires_at and 
            user.verification_code_expires_at > datetime.datetime.now()):
            
            user.is_verified = True
            user.verification_code = None
            user.verification_code_expires_at = None
            db.commit()
            db.refresh(user)
            return True
            
        return False


user = CRUDUser(User) 