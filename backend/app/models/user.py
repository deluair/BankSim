"""
SQLAlchemy model for User.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Table, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..db.session import Base # Adjusted import path

# Association table for many-to-many relationship between Users and Roles
user_roles_table = Table('user_roles', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False) # For admin privileges
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    roles = relationship("Role", secondary=user_roles_table, back_populates="users")
    # Add other relationships here if needed, e.g., simulations created by user
    # simulations = relationship("SimulationRun", back_populates="owner")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(String(255), nullable=True)

    users = relationship("User", secondary=user_roles_table, back_populates="roles")

    def __repr__(self):
        return f"<Role(id={self.id}, name='{self.name}')>"

# You might want to pre-populate some roles, e.g., in your init_db function or a separate script
# DEFAULT_ROLES = [
#     {"name": "admin", "description": "Administrator with full access"},
#     {"name": "policy_analyst", "description": "User who can run simulations and view results"},
#     {"name": "researcher", "description": "User with access to data and basic simulation tools"},
#     {"name": "guest", "description": "User with limited read-only access"}
# ]
