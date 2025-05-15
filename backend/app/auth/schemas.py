"""
Pydantic schemas for authentication and user management.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class Token(BaseModel):
    """Schema for JWT access token."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for data encoded in JWT token."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: List[str] = []

class UserBase(BaseModel):
    """Base schema for user data."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8)

class UserInDBBase(UserBase):
    """Base schema for user data stored in DB."""
    id: int

    class Config:
        orm_mode = True # Changed from from_attributes for Pydantic v2

class User(UserInDBBase):
    """Schema for returning user data."""
    pass

class UserInDB(UserInDBBase):
    """Schema for user data including hashed password (for internal use)."""
    hashed_password: str

class PasswordResetRequest(BaseModel):
    """Schema for requesting a password reset."""
    email: EmailStr

class PasswordReset(BaseModel):
    """Schema for resetting a password."""
    token: str
    new_password: str = Field(..., min_length=8)
