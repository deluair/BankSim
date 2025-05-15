"""
Security utility functions for password hashing, token creation, and verification.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Any, List

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# --- Configuration ---
# These should be moved to a configuration file or environment variables in a real application
SECRET_KEY = "your-secret-key-for-jwt-Hs256-should-be-long-and-random"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password Hashing Context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Token Schemas (can be in schemas.py or here if specific to security layer) ---
class TokenPayload(BaseModel):
    sub: Union[str, int] # Subject of the token (e.g., username or user_id)
    exp: datetime
    scopes: list[str] = []
    # Add any other claims you need, e.g., user_id, role
    user_id: Optional[int] = None
    username: Optional[str] = None

# --- Password Utilities ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a plain password."""
    return pwd_context.hash(password)

# --- JWT Token Utilities ---

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create a new JWT refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM) # Can use a different secret/algo for refresh tokens
    return encoded_jwt

def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode a JWT token and return its payload if valid."""
    try:
        payload_dict = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenPayload(**payload_dict)
    except JWTError:
        return None
    except Exception as e:
        # Log the error for debugging
        print(f"Error decoding token: {e}")
        return None

# Example of creating token payload more explicitly
def create_token_payload(subject: Union[str, Any], user_id: Optional[int] = None, username: Optional[str] = None, scopes: list[str] = None) -> dict:
    """Helper to create a consistent token payload."""
    if scopes is None:
        scopes = []
    payload = {"sub": str(subject), "scopes": scopes}
    if user_id is not None:
        payload["user_id"] = user_id
    if username is not None:
        payload["username"] = username
    return payload

# --- Scope/Permission Utilities (Example) ---

def check_scope(required_scope: str, token_scopes: List[str]) -> bool:
    """Check if a required scope is present in the token's scopes."""
    # This is a simple check. You might have more complex logic,
    # e.g., hierarchical scopes or checking against user roles.
    return required_scope in token_scopes
