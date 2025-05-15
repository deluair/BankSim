"""
CRUD operations for User model.
"""

from typing import Optional, List, Union
from sqlalchemy.orm import Session

from . import schemas, security # Use relative imports within the 'auth' package
from ..models.user import User, Role # Adjusted import for User and Role models

# --- User CRUD --- 

def get_user(db: Session, user_id: int) -> Optional[User]:
    """Retrieve a user by their ID."""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieve a user by their email address."""
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Retrieve a user by their username."""
    return db.query(User).filter(User.username == username).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Retrieve a list of users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate) -> User:
    """Create a new user in the database."""
    hashed_password = security.get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        is_active=user.is_active if user.is_active is not None else True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(
    db: Session,
    user_id: int,
    user_update: Union[schemas.UserBase, dict]
) -> Optional[User]:
    """Update an existing user's information."""
    db_user = get_user(db, user_id)
    if not db_user:
        return None

    update_data = user_update if isinstance(user_update, dict) else user_update.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if key == "password" and value is not None: # Handle password update separately
            hashed_password = security.get_password_hash(value)
            setattr(db_user, "hashed_password", hashed_password)
        elif hasattr(db_user, key):
            setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

def activate_deactivate_user(db: Session, user_id: int, is_active: bool) -> Optional[User]:
    """Activate or deactivate a user."""
    db_user = get_user(db, user_id)
    if db_user:
        db_user.is_active = is_active
        db.commit()
        db.refresh(db_user)
    return db_user


# --- Role CRUD --- (Basic Role Management)

def get_role_by_name(db: Session, name: str) -> Optional[Role]:
    """Retrieve a role by its name."""
    return db.query(Role).filter(Role.name == name).first()

def create_role(db: Session, role_name: str, description: Optional[str] = None) -> Role:
    """Create a new role."""
    db_role = Role(name=role_name, description=description)
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    return db_role

def get_roles(db: Session, skip: int = 0, limit: int = 20) -> List[Role]:
    """Retrieve a list of roles."""
    return db.query(Role).offset(skip).limit(limit).all()

def assign_role_to_user(db: Session, user_id: int, role_name: str) -> Optional[User]:
    """Assign a role to a user."""
    user = get_user(db, user_id)
    role = get_role_by_name(db, role_name)
    if user and role:
        if role not in user.roles: # Avoid duplicate assignments
            user.roles.append(role)
            db.commit()
            db.refresh(user)
        return user
    return None

def remove_role_from_user(db: Session, user_id: int, role_name: str) -> Optional[User]:
    """Remove a role from a user."""
    user = get_user(db, user_id)
    role = get_role_by_name(db, role_name)
    if user and role and role in user.roles:
        user.roles.remove(role)
        db.commit()
        db.refresh(user)
        return user
    return None

# Helper to initialize default roles if they don't exist
def initialize_default_roles(db: Session):
    """Creates default roles in the database if they don't already exist."""
    DEFAULT_ROLES = [
        {"name": "admin", "description": "Administrator with full access"},
        {"name": "policy_analyst", "description": "User who can run simulations and view results"},
        {"name": "researcher", "description": "User with access to data and basic simulation tools"},
        {"name": "guest", "description": "User with limited read-only access"}
    ]
    for role_data in DEFAULT_ROLES:
        role = get_role_by_name(db, role_data["name"])
        if not role:
            create_role(db, role_name=role_data["name"], description=role_data["description"])
            print(f"Created role: {role_data['name']}")
