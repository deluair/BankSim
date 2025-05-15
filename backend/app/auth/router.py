"""
API router for authentication and user management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Any

from . import schemas, security, crud_user
from ..db.session import get_db
from ..models.user import User as DBUser # Alias to avoid Pydantic/SQLAlchemy model name clash

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)

# OAuth2PasswordBearer is used to get the token from the request's Authorization header
# tokenUrl should point to your token endpoint (this router's /token path)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# --- Helper function to get current user ---
async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> DBUser:
    """Decode token, validate, and return the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = security.decode_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception
    
    user = crud_user.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

async def get_current_active_user(
    current_user: DBUser = Depends(get_current_user)
) -> DBUser:
    """Ensure the current user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_active_superuser(
    current_user: DBUser = Depends(get_current_active_user),
) -> DBUser:
    """Ensure the current user is an active superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user

# --- API Endpoints ---

@router.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Authenticate user and return an access token.
    Uses OAuth2PasswordRequestForm for username/password input (standard for token endpoints).
    """
    user = crud_user.get_user_by_username(db, username=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # Add scopes if you implement them
    # For now, an empty list or default scopes
    scopes = ["me"] # Example scope
    if user.is_superuser:
        scopes.append("admin")
        
    # Create token payload
    token_payload_data = security.create_token_payload(
        subject=user.username, 
        user_id=user.id,
        username=user.username,
        scopes=scopes
    )
    access_token = security.create_access_token(data=token_payload_data)
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
async def create_new_user(
    user_in: schemas.UserCreate,
    db: Session = Depends(get_db)
    # current_user: DBUser = Depends(get_current_active_superuser) # Uncomment to restrict creation to superusers
):
    """
    Create a new user. 
    Currently open, but typically this would be restricted (e.g., to superusers or via an invite system).
    """
    db_user_by_email = crud_user.get_user_by_email(db, email=user_in.email)
    if db_user_by_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered."
        )
    db_user_by_username = crud_user.get_user_by_username(db, username=user_in.username)
    if db_user_by_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken."
        )
    
    created_user = crud_user.create_user(db=db, user=user_in)
    # Assign a default role, e.g., 'guest' or 'policy_analyst'
    # crud_user.assign_role_to_user(db, user_id=created_user.id, role_name="policy_analyst")
    return created_user


@router.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: DBUser = Depends(get_current_active_user)):
    """Get current logged-in user's details."""
    return current_user


@router.get("/users/{user_id}", response_model=schemas.User)
async def read_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_superuser) # Requires superuser to view arbitrary users
):
    """Get a specific user by ID (superuser access)."""
    db_user = crud_user.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/users/", response_model=List[schemas.User])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_superuser) # Requires superuser
):
    """Retrieve a list of users (superuser access)."""
    users = crud_user.get_users(db, skip=skip, limit=limit)
    return users


@router.put("/users/me", response_model=schemas.User)
async def update_user_me(
    user_in: schemas.UserBase, # Allows updating basic fields, not password here
    db: Session = Depends(get_db),
    current_user: DBUser = Depends(get_current_active_user)
):
    """Update current logged-in user's details."""
    # Ensure email is not taken by another user if it's being changed
    if user_in.email and user_in.email != current_user.email:
        existing_user = crud_user.get_user_by_email(db, email=user_in.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered by another user.")
    
    # Ensure username is not taken by another user if it's being changed
    if user_in.username and user_in.username != current_user.username:
        existing_user = crud_user.get_user_by_username(db, username=user_in.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken by another user.")

    updated_user = crud_user.update_user(db, user_id=current_user.id, user_update=user_in)
    return updated_user

# Add more endpoints as needed: e.g., password change, role assignment, etc.
