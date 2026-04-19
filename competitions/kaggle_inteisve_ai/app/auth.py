"""Authentication utilities for EmergencyCareNavigator."""
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from app.models import User, LoginRequest, RegisterRequest
from app.database import SessionLocal, UserModel

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# User storage - now using database instead of JSON files
# JSON file path kept for backward compatibility/migration
if os.getenv("VERCEL"):
    USERS_FILE = "/tmp/users.json"
else:
    USERS_FILE = "users.json"


def hash_password(password: str) -> str:
    """Hash password using SHA256 (simple hashing for demo)."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == password_hash


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email from database."""
    # CRITICAL: Ensure database is initialized before any query
    try:
        from app.database import init_db, SessionLocal, UserModel
        init_db()  # This is safe to call multiple times (thread-safe)
    except Exception as init_error:
        import logging
        logging.error(f"Failed to import/initialize database in get_user_by_email: {init_error}", exc_info=True)
        return None
    
    db = None
    try:
        db = SessionLocal()
        user_model = db.query(UserModel).filter(UserModel.email == email.lower()).first()
        if not user_model:
            return None
        
        return User(
            id=user_model.id,
            email=user_model.email,
            name=user_model.name,
            role=user_model.role,
            facility_name=user_model.facility_name,
            password_hash=user_model.password_hash
        )
    except Exception as e:
        import logging
        logging.error(f"Error getting user by email {email}: {e}", exc_info=True)
        return None
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


def create_user(email: str, password: str, name: str, role: str, facility_name: Optional[str] = None) -> User:
    """Create a new user in database."""
    # CRITICAL: Ensure database is initialized before any query
    from app.database import init_db
    init_db()  # This is safe to call multiple times (thread-safe)
    
    db = SessionLocal()
    try:
        # Check if user already exists
        existing = db.query(UserModel).filter(UserModel.email == email.lower()).first()
        if existing:
            raise ValueError("User already exists")
        
        if role == "hospital_staff" and not facility_name:
            raise ValueError("facility_name is required for hospital staff")
        
        # Create new user
        user_model = UserModel(
            id=secrets.token_urlsafe(16),
            email=email.lower(),
            name=name,
            role=role,
            facility_name=facility_name,
            password_hash=hash_password(password)
        )
        
        db.add(user_model)
        db.commit()
        db.refresh(user_model)
        
        return User(
            id=user_model.id,
            email=user_model.email,
            name=user_model.name,
            role=user_model.role,
            facility_name=user_model.facility_name,
            password_hash=user_model.password_hash
        )
    except ValueError:
        raise
    except Exception as e:
        db.rollback()
        import logging
        logging.error(f"Failed to create user {email}: {e}")
        raise ValueError(f"Failed to create user: {str(e)}")
    finally:
        db.close()


def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user and return User if valid."""
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# Initialize with demo users if they don't exist in database
def init_demo_users():
    """Initialize demo users for testing in database."""
    # CRITICAL: Ensure database is initialized before any query
    from app.database import init_db
    init_db()  # This is safe to call multiple times (thread-safe)
    
    db = SessionLocal()
    try:
        # Check if any users exist
        user_count = db.query(UserModel).count()
        if user_count == 0:
            # Demo patient
            try:
                create_user("patient@demo.com", "patient123", "Demo Patient", "patient")
            except ValueError:
                pass  # User already exists
            
            # Demo hospital staff
            try:
                create_user("hospital@demo.com", "hospital123", "Hospital Staff", "hospital_staff", "Aga Khan University Hospital")
            except ValueError:
                pass  # User already exists
            
            try:
                create_user("staff@demo.com", "staff123", "Receptionist", "hospital_staff", "Jinnah Hospital")
            except ValueError:
                pass  # User already exists
    except Exception as e:
        import logging
        logging.warning(f"Failed to initialize demo users: {e}")
    finally:
        db.close()


# Initialize on import (will be called on startup)
# Note: This runs on module import, but database might not be initialized yet
# So we also call it in the startup event
try:
    init_demo_users()
except Exception:
    pass  # Will be initialized in startup event





