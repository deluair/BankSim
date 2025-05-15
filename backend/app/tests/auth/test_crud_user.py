import pytest
from sqlalchemy.orm import Session
from app.auth import crud_user, schemas as auth_schemas
from app.models import user as user_models # To access User and Role models directly for setup
from app.db.session import SessionLocal, engine # Assuming you have these for test DB setup

# Helper to initialize a test database (if not using a global test DB setup)
# This is a simplified setup. In a real app, you'd use a fixture for DB sessions.
def setup_test_db():
    user_models.Base.metadata.create_all(bind=engine)

def teardown_test_db():
    user_models.Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session() -> Session:
    setup_test_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        teardown_test_db() # Clean up after each test function

@pytest.fixture(scope="function")
def test_role_admin(db_session: Session) -> user_models.Role:
    role_in = auth_schemas.RoleCreate(name="admin_test_role", description="Test Admin Role")
    return crud_user.create_role(db_session, role_in)

@pytest.fixture(scope="function")
def test_role_user(db_session: Session) -> user_models.Role:
    role_in = auth_schemas.RoleCreate(name="user_test_role", description="Test User Role")
    return crud_user.create_role(db_session, role_in)


def test_create_and_get_user(db_session: Session):
    user_in = auth_schemas.UserCreate(
        username="testuser1", 
        email="testuser1@example.com", 
        password="testpassword123"
    )
    db_user = crud_user.create_user(db=db_session, user=user_in)
    assert db_user is not None
    assert db_user.username == user_in.username
    assert db_user.email == user_in.email
    assert db_user.hashed_password != user_in.password # Ensure password was hashed

    retrieved_user_by_email = crud_user.get_user_by_email(db=db_session, email=user_in.email)
    assert retrieved_user_by_email is not None
    assert retrieved_user_by_email.id == db_user.id

    retrieved_user_by_username = crud_user.get_user_by_username(db=db_session, username=user_in.username)
    assert retrieved_user_by_username is not None
    assert retrieved_user_by_username.id == db_user.id

    retrieved_user_by_id = crud_user.get_user(db=db_session, user_id=db_user.id)
    assert retrieved_user_by_id is not None
    assert retrieved_user_by_id.id == db_user.id


def test_update_user(db_session: Session):
    user_in = auth_schemas.UserCreate(username="updateuser", email="update@example.com", password="pass1")
    db_user = crud_user.create_user(db=db_session, user=user_in)

    user_update_data = auth_schemas.UserUpdate(email="updated_email@example.com", username="updated_username")
    updated_user = crud_user.update_user(db=db_session, user_id=db_user.id, user_update=user_update_data)
    
    assert updated_user is not None
    assert updated_user.email == "updated_email@example.com"
    assert updated_user.username == "updated_username"
    # Check that other fields like password hash are not accidentally changed if not specified
    assert updated_user.hashed_password == db_user.hashed_password 

    # Test partial update (only email)
    user_partial_update = auth_schemas.UserUpdate(email="partialupdate@example.com")
    partially_updated_user = crud_user.update_user(db=db_session, user_id=db_user.id, user_update=user_partial_update)
    assert partially_updated_user.email == "partialupdate@example.com"
    assert partially_updated_user.username == "updated_username" # Username should remain from previous update


def test_delete_user(db_session: Session):
    user_in = auth_schemas.UserCreate(username="deleteuser", email="delete@example.com", password="passdel")
    db_user = crud_user.create_user(db=db_session, user=user_in)
    assert db_user is not None

    deleted_user = crud_user.delete_user(db=db_session, user_id=db_user.id)
    assert deleted_user is not None
    assert deleted_user.id == db_user.id

    assert crud_user.get_user(db=db_session, user_id=db_user.id) is None


def test_create_and_get_role(db_session: Session):
    role_in = auth_schemas.RoleCreate(name="testrole", description="A test role for CRUD")
    db_role = crud_user.create_role(db=db_session, role=role_in)
    assert db_role is not None
    assert db_role.name == role_in.name
    assert db_role.description == role_in.description

    retrieved_role = crud_user.get_role_by_name(db=db_session, name=role_in.name)
    assert retrieved_role is not None
    assert retrieved_role.id == db_role.id


def test_assign_and_remove_role_from_user(db_session: Session, test_role_admin: user_models.Role):
    user_in = auth_schemas.UserCreate(username="rolassignuser", email="roleassign@example.com", password="passrole")
    db_user = crud_user.create_user(db=db_session, user=user_in)
    assert db_user is not None
    assert test_role_admin is not None

    # Assign role
    user_with_role = crud_user.assign_role_to_user(db=db_session, user_id=db_user.id, role_id=test_role_admin.id)
    assert user_with_role is not None
    assert any(role.id == test_role_admin.id for role in user_with_role.roles)

    # Check if role is actually in user's roles list
    db_user_reloaded = crud_user.get_user(db=db_session, user_id=db_user.id)
    assert any(role.id == test_role_admin.id for role in db_user_reloaded.roles)

    # Remove role
    user_without_role = crud_user.remove_role_from_user(db=db_session, user_id=db_user.id, role_id=test_role_admin.id)
    assert user_without_role is not None
    assert not any(role.id == test_role_admin.id for role in user_without_role.roles)

    # Check if role is actually removed
    db_user_reloaded_again = crud_user.get_user(db=db_session, user_id=db_user.id)
    assert not any(role.id == test_role_admin.id for role in db_user_reloaded_again.roles)


def test_get_non_existent_user(db_session: Session):
    user = crud_user.get_user(db=db_session, user_id=99999)
    assert user is None
    user_by_email = crud_user.get_user_by_email(db=db_session, email="nonexistent@example.com")
    assert user_by_email is None
    user_by_username = crud_user.get_user_by_username(db=db_session, username="nonexistentuser")
    assert user_by_username is None


def test_get_non_existent_role(db_session: Session):
    role = crud_user.get_role_by_name(db=db_session, name="nonexistentrole")
    assert role is None


# To run these tests, navigate to the backend directory and run:
# pytest app/tests/auth/test_crud_user.py
# Ensure your test database is configured correctly (e.g., via environment variables or a test config)
