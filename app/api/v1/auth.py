from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.core.config import settings
from app.core.security import create_access_token
from app.core.deps import get_current_user
from app.crud.crud_user import user as crud_user
from app.models.user import User
from app.schemas.user import User as UserSchema, UserCreate
from app.schemas.token import Token
from app.db.base import get_db
from app.services.email import email_service

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


@router.post("/login", response_model=Token)
def login_access_token(
    db: Session = Depends(get_db), credentials: LoginRequest = Body(...)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = crud_user.authenticate(
        db, email=credentials.username, password=credentials.password
    )
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not crud_user.is_active(user):
        raise HTTPException(status_code=400, detail="Inactive user")
    # Check if user is verified
    elif not user.is_verified:
        raise HTTPException(status_code=400, detail="Email not verified. Please verify your email before logging in.")
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }


@router.post("/register", response_model=UserSchema)
def create_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Create new user without the need to be logged in.
    """
    user = crud_user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    username_exists = crud_user.get_by_username(db, username=user_in.username)
    if username_exists:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )
    
    # Create user with verification code
    user = crud_user.create(db, obj_in=user_in)
    
    # Send verification email
    email_service.send_verification_email(user.email, user.verification_code)
    
    return user


@router.post("/verify-email")
def verify_email(
    *,
    db: Session = Depends(get_db),
    verification_data: VerifyEmailRequest,
) -> Any:
    """
    Verify a user's email with the verification code
    """
    user = crud_user.get_by_email(db, email=verification_data.email)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    
    if user.is_verified:
        return {"message": "Email already verified"}
    
    if crud_user.verify_email(db, email=verification_data.email, code=verification_data.code):
        return {"message": "Email verified successfully"}
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired verification code",
        )


@router.post("/resend-verification")
def resend_verification(
    *,
    db: Session = Depends(get_db),
    resend_data: ResendVerificationRequest,
) -> Any:
    """
    Resend verification code to a user's email
    """
    user = crud_user.get_by_email(db, email=resend_data.email)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    
    if user.is_verified:
        return {"message": "Email already verified"}
    
    # Generate new verification code
    verification_code = crud_user.set_verification_code(db, user_id=user.id)
    
    # Send verification email
    email_service.send_verification_email(user.email, verification_code)
    
    return {"message": "Verification code resent successfully"}


@router.get("/me", response_model=UserSchema)
def read_users_me(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get current user.
    """
    return current_user 