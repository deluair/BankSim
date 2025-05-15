import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.main import app  # The main FastAPI application
from app.db.session import Base, get_db
from app.auth import schemas as auth_schemas, crud_user
from app.models.user import User as DBUser # SQLAlchemy model

# --- Test Database Setup ---
# Using an in-memory SQLite database for testing
#SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth_router.db" # For a file-based test DB
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}, # Needed only for SQLite
    poolclass=StaticPool, # Use StaticPool for in-memory SQLite in tests
)

# Override the get_db dependency for testing
def override_get_db():
    db = None
    try:
        Base.metadata.create_all(bind=engine) # Create tables for each test session
        db = Session(engine)
        yield db
    finally:
        if db:
            db.close()
        Base.metadata.drop_all(bind=engine) # Drop tables after each test session

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# --- Fixtures & Helper Functions ---

@pytest.fixture(scope="function")
def db_session_for_router(override_get_db):
    # This fixture ensures the database is clean for each test function
    # by leveraging the overridden get_db which handles setup/teardown.
    # We can obtain a session directly if needed for setup, but TestClient will use the override.
    # For direct DB manipulation in tests:
    session_gen = override_get_db()
    db = next(session_gen)
    yield db
    try:
        next(session_gen) # This will trigger the finally block in override_get_db
    except StopIteration:
        pass

def create_test_user_direct_db(db: Session, user_data: auth_schemas.UserCreate, is_superuser: bool = False) -> DBUser:
    """Helper to create a user directly in DB for test setup."""
    db_user = crud_user.create_user(db, user=user_data)
    if is_superuser:
        db_user.is_superuser = True
        db.commit()
        db.refresh(db_user)
    return db_user

# --- Test Cases ---

TEST_USER_USERNAME = "testuser_router"
TEST_USER_EMAIL = "testuser_router@example.com"
TEST_USER_PASSWORD = "testpassword123"

TEST_SUPERUSER_USERNAME = "super_router"
TEST_SUPERUSER_EMAIL = "super_router@example.com"
TEST_SUPERUSER_PASSWORD = "superpassword123"


def test_create_user_endpoint(db_session_for_router: Session):
    response = client.post(
        "/auth/users/",
        json={"username": "newuser", "email": "newuser@example.com", "password": "newpass123"}
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert "id" in data
    assert data["is_active"] == True
    assert data["is_superuser"] == False
    assert "hashed_password" not in data # Ensure password is not returned

    # Test duplicate username
    response_dup_username = client.post(
        "/auth/users/",
        json={"username": "newuser", "email": "another@example.com", "password": "newpass123"}
    )
    assert response_dup_username.status_code == 400
    assert "Username already taken" in response_dup_username.text

    # Test duplicate email
    response_dup_email = client.post(
        "/auth/users/",
        json={"username": "anotheruser", "email": "newuser@example.com", "password": "newpass123"}
    )
    assert response_dup_email.status_code == 400
    assert "Email already registered" in response_dup_email.text

def test_login_for_access_token(db_session_for_router: Session):
    # Create a user first
    user_in = auth_schemas.UserCreate(username=TEST_USER_USERNAME, email=TEST_USER_EMAIL, password=TEST_USER_PASSWORD)
    create_test_user_direct_db(db_session_for_router, user_in)

    login_data = {"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD}
    response = client.post("/auth/token", data=login_data) # form data
    assert response.status_code == 200, response.text
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

    # Test login with wrong password
    wrong_login_data = {"username": TEST_USER_USERNAME, "password": "wrongpassword"}
    response_wrong_pass = client.post("/auth/token", data=wrong_login_data)
    assert response_wrong_pass.status_code == 401
    assert "Incorrect username or password" in response_wrong_pass.text

    # Test login with non-existent user
    non_existent_login = {"username": "nosuchuser", "password": "anypassword"}
    response_no_user = client.post("/auth/token", data=non_existent_login)
    assert response_no_user.status_code == 401

def test_read_users_me(db_session_for_router: Session):
    user_in = auth_schemas.UserCreate(username=TEST_USER_USERNAME, email=TEST_USER_EMAIL, password=TEST_USER_PASSWORD)
    create_test_user_direct_db(db_session_for_router, user_in)

    login_response = client.post("/auth/token", data={"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response_me = client.get("/auth/users/me", headers=headers)
    assert response_me.status_code == 200, response_me.text
    data = response_me.json()
    assert data["username"] == TEST_USER_USERNAME
    assert data["email"] == TEST_USER_EMAIL

    # Test without token
    response_no_token = client.get("/auth/users/me")
    assert response_no_token.status_code == 401 # Expecting unauthorized

def test_update_user_me(db_session_for_router: Session):
    user_in = auth_schemas.UserCreate(username=TEST_USER_USERNAME, email=TEST_USER_EMAIL, password=TEST_USER_PASSWORD)
    create_test_user_direct_db(db_session_for_router, user_in)

    login_response = client.post("/auth/token", data={"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    update_data = {"full_name": "Test User Router Updated", "email": "updated_router@example.com"}
    response_update = client.put("/auth/users/me", headers=headers, json=update_data)
    assert response_update.status_code == 200, response_update.text
    updated_user_data = response_update.json()
    assert updated_user_data["full_name"] == "Test User Router Updated"
    assert updated_user_data["email"] == "updated_router@example.com"

    # Try to update username (should work if not conflicting)
    username_update_data = {"username": "new_test_router_username"}
    response_username_update = client.put("/auth/users/me", headers=headers, json=username_update_data)
    assert response_username_update.status_code == 200, response_username_update.text
    assert response_username_update.json()["username"] == "new_test_router_username"

    # Create another user to test email/username conflict on update
    other_user_data = auth_schemas.UserCreate(username="otheruser", email="other@example.com", password="otherpass")
    create_test_user_direct_db(db_session_for_router, other_user_data)

    conflict_update_data = {"email": "other@example.com"} # Email taken by otheruser
    response_conflict = client.put("/auth/users/me", headers=headers, json=conflict_update_data)
    assert response_conflict.status_code == 400
    assert "Email already registered" in response_conflict.text

# --- Superuser restricted endpoint tests ---

@pytest.fixture(scope="function")
def superuser_token_headers(db_session_for_router: Session) -> dict:
    superuser_in = auth_schemas.UserCreate(username=TEST_SUPERUSER_USERNAME, email=TEST_SUPERUSER_EMAIL, password=TEST_SUPERUSER_PASSWORD)
    create_test_user_direct_db(db_session_for_router, superuser_in, is_superuser=True)
    
    login_data = {"username": TEST_SUPERUSER_USERNAME, "password": TEST_SUPERUSER_PASSWORD}
    response = client.post("/auth/token", data=login_data)
    assert response.status_code == 200, f"Failed to get superuser token: {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(scope="function")
def normal_user_token_headers(db_session_for_router: Session) -> dict:
    user_in = auth_schemas.UserCreate(username=TEST_USER_USERNAME, email=TEST_USER_EMAIL, password=TEST_USER_PASSWORD)
    create_test_user_direct_db(db_session_for_router, user_in, is_superuser=False)
    
    login_data = {"username": TEST_USER_USERNAME, "password": TEST_USER_PASSWORD}
    response = client.post("/auth/token", data=login_data)
    assert response.status_code == 200, f"Failed to get normal user token: {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_read_users_superuser(db_session_for_router: Session, superuser_token_headers: dict):
    # Create a couple of users to list
    auth_schemas.UserCreate(username="user1_list", email="user1_list@example.com", password="pass1")
    auth_schemas.UserCreate(username="user2_list", email="user2_list@example.com", password="pass2")
    # Note: these users are not created directly in this test's scope, relying on the db_session to be clean or creating them.
    # For more robust test, explicitly create them using create_test_user_direct_db if needed for count.
    # The superuser itself will also be in the list.

    response = client.get("/auth/users/", headers=superuser_token_headers)
    assert response.status_code == 200, response.text
    users_list = response.json()
    assert isinstance(users_list, list)
    # Check if the superuser who made the request is in the list (it should be)
    assert any(user["username"] == TEST_SUPERUSER_USERNAME for user in users_list)

def test_read_users_normal_user_forbidden(db_session_for_router: Session, normal_user_token_headers: dict):
    response = client.get("/auth/users/", headers=normal_user_token_headers)
    assert response.status_code == 403 # Forbidden for non-superusers

def test_read_user_by_id_superuser(db_session_for_router: Session, superuser_token_headers: dict):
    # Create a user to fetch
    user_to_fetch_data = auth_schemas.UserCreate(username="fetchme", email="fetchme@example.com", password="fetchpass")
    user_to_fetch = create_test_user_direct_db(db_session_for_router, user_to_fetch_data)

    response = client.get(f"/auth/users/{user_to_fetch.id}", headers=superuser_token_headers)
    assert response.status_code == 200, response.text
    fetched_user = response.json()
    assert fetched_user["username"] == "fetchme"

    # Test fetching non-existent user
    response_not_found = client.get(f"/auth/users/99999", headers=superuser_token_headers)
    assert response_not_found.status_code == 404

def test_read_user_by_id_normal_user_forbidden(db_session_for_router: Session, normal_user_token_headers: dict):
    # Create a user (doesn't matter which, normal user can't access this endpoint anyway)
    user_data = auth_schemas.UserCreate(username="anyuser", email="any@example.com", password="anypass")
    target_user = create_test_user_direct_db(db_session_for_router, user_data)

    response = client.get(f"/auth/users/{target_user.id}", headers=normal_user_token_headers)
    assert response.status_code == 403 # Forbidden

# Add more tests: e.g., inactive user login, token expiry (might need to mock time or adjust token life for tests)
