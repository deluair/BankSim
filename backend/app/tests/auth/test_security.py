import pytest
from jose import jwt
from datetime import timedelta, datetime, timezone

from app.auth.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    SECRET_KEY,
    create_access_token,
    get_password_hash,
    verify_password,
    decode_token
)


def test_password_hashing():
    password = "plainpassword"
    hashed_password = get_password_hash(password)
    assert hashed_password != password
    assert verify_password(password, hashed_password)
    assert not verify_password("wrongpassword", hashed_password)


def test_create_and_decode_access_token():
    user_identifier = "testuser@example.com"
    data = {"sub": user_identifier}
    token = create_access_token(data)
    
    decoded_payload = decode_token(token)
    assert decoded_payload is not None
    assert decoded_payload.get("sub") == user_identifier
    assert "exp" in decoded_payload

    # Test with custom expiry
    expires_delta = timedelta(minutes=15)
    token_custom_expiry = create_access_token(data, expires_delta=expires_delta)
    decoded_payload_custom_expiry = decode_token(token_custom_expiry)
    assert decoded_payload_custom_expiry is not None
    assert decoded_payload_custom_expiry.get("sub") == user_identifier


def test_decode_token_invalid_token_format():
    invalid_token = "this.is.an.invalid.token.format"
    payload = decode_token(invalid_token)
    assert payload is None


def test_decode_token_tampered_signature():
    data = {"sub": "testuser_tampered@example.com"}
    token_parts = create_access_token(data).split('.')
    # Simulate tampering by slightly altering the signature part
    if len(token_parts) == 3:
        tampered_signature = token_parts[2][:-1] + ('a' if token_parts[2][-1] != 'a' else 'b')
        tampered_token = f"{token_parts[0]}.{token_parts[1]}.{tampered_signature}"
        payload = decode_token(tampered_token)
        assert payload is None
    else:
        # Fallback if token format is unexpected, though create_access_token should be consistent
        pytest.skip("Token format not as expected for tampering test")


def test_decode_token_expired():
    user_identifier = "testuser_expired@example.com"
    data = {"sub": user_identifier}
    
    # Create a token that is already expired
    # The simplest way is to provide a negative timedelta for expiry
    expired_delta = timedelta(seconds=-ACCESS_TOKEN_EXPIRE_MINUTES * 60 -1) # Ensure it's definitely expired
    
    # To create an effectively expired token, we encode it with an expiry time in the past.
    expire = datetime.now(timezone.utc) + expired_delta
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    expired_token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    payload = decode_token(expired_token)
    assert payload is None, "Token should be considered expired"


# To run these tests, navigate to the `backend` directory and run:
# pytest
# Or specifically for this file:
# pytest app/tests/auth/test_security.py
