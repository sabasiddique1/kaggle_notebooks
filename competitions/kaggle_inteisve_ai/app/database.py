"""Database setup and connection."""
import os
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
    from datetime import datetime
    import uuid
    logger.info("SQLAlchemy imports successful")
except ImportError as e:
    logger.error(f"Failed to import SQLAlchemy: {e}", exc_info=True)
    raise

# Database configuration
# Production: Use PostgreSQL (via DATABASE_URL or Supabase env vars)
# Local: Use SQLite for development
# 
# Setup options:
# 1. EASIEST: Set SUPABASE_URL and SUPABASE_DB_PASSWORD (we'll build connection string)
# 2. ALTERNATIVE: Set DATABASE_URL directly (postgresql://user:password@host:port/database)
# 3. LOCAL: Don't set anything - uses SQLite automatically

DATABASE_URL = os.getenv("DATABASE_URL")

# If DATABASE_URL not set, try building from Supabase env vars
if not DATABASE_URL:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_password = os.getenv("SUPABASE_DB_PASSWORD")
    
    if supabase_url and supabase_password:
        # Extract project reference from Supabase URL
        # Format: https://[PROJECT_REF].supabase.co
        # Or: https://lgbpmgaacqawvfavtdzu.supabase.co
        try:
            # Remove https:// and .supabase.co to get project ref
            # Handle both with and without trailing slash
            clean_url = supabase_url.strip().rstrip('/')
            if clean_url.startswith("https://"):
                clean_url = clean_url.replace("https://", "")
            if clean_url.endswith(".supabase.co"):
                project_ref = clean_url.replace(".supabase.co", "")
            else:
                # If format is different, try to extract project ref
                parts = clean_url.split('.')
                if len(parts) >= 2:
                    project_ref = parts[0]
                else:
                    raise ValueError(f"Invalid Supabase URL format: {supabase_url}")
            
            # Build PostgreSQL connection string
            DATABASE_URL = f"postgresql://postgres:{supabase_password}@db.{project_ref}.supabase.co:5432/postgres"
            import logging
            logging.info(f"Built DATABASE_URL from Supabase env vars (project: {project_ref})")
        except Exception as e:
            import logging
            logging.error(f"Failed to build DATABASE_URL from Supabase vars: {e}", exc_info=True)
            # Don't set DATABASE_URL - let it fall back to SQLite or error

# If still no DATABASE_URL, use SQLite
if not DATABASE_URL:
    if os.getenv("VERCEL"):
        # On Vercel without DATABASE_URL, use /tmp (but this is unreliable)
        DATABASE_URL = "sqlite:////tmp/emergencycare.db"
    else:
        # Local development
        DATABASE_URL = "sqlite:///./emergencycare.db"

# Convert postgres:// to postgresql:// for SQLAlchemy compatibility
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate connection args
if "sqlite" in DATABASE_URL:
    # SQLite-specific connection args
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True  # Verify connections before using
    )
else:
    # PostgreSQL connection args
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,  # Connection pool size
        max_overflow=10  # Max overflow connections
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Database Models
class UserModel(Base):
    """User table for authentication."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "patient" or "hospital"
    facility_name = Column(String, nullable=True)  # For hospital staff
    created_at = Column(DateTime, default=datetime.utcnow)


class TriageSessionModel(Base):
    """Triage session table - stores complete session data."""
    __tablename__ = "triage_sessions"
    
    session_id = Column(String, primary_key=True)
    patient_email = Column(String, index=True, nullable=True)
    patient_name = Column(String, nullable=True)
    
    # Intake data (stored as JSON)
    intake_data = Column(JSON, nullable=True)
    
    # Triage result
    triage_level = Column(String, nullable=True)
    triage_reason = Column(Text, nullable=True)
    triage_action = Column(Text, nullable=True)
    triage_safety_note = Column(Text, nullable=True)
    
    # Location
    location_query = Column(String, nullable=True)
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)
    
    # Recommendation data (stored as JSON)
    recommendation_data = Column(JSON, nullable=True)
    
    # Booking state
    request_type = Column(String, nullable=True)  # "alert" or "appointment"
    booking_status = Column(String, nullable=True)
    facility_name = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    bookings = relationship("BookingModel", back_populates="session", cascade="all, delete-orphan")


class BookingModel(Base):
    """Booking/Request table - stores hospital bookings and patient requests."""
    __tablename__ = "bookings"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("triage_sessions.session_id"), index=True, nullable=False)
    
    # Patient info
    patient_email = Column(String, index=True, nullable=True)
    patient_name = Column(String, nullable=False)
    
    # Facility info
    facility_name = Column(String, index=True, nullable=False)
    
    # Triage info
    triage_level = Column(String, nullable=False)
    request_type = Column(String, nullable=False)  # "alert" or "appointment"
    
    # Status
    status = Column(String, nullable=False)  # PENDING_ACK, ACKNOWLEDGED, PENDING_APPROVAL, CONFIRMED, REJECTED
    
    # Handoff packet
    handoff_packet = Column(Text, nullable=True)
    
    # ETA and location
    eta_minutes = Column(Integer, nullable=True)
    location = Column(String, nullable=True)
    
    # Symptoms (stored as JSON array)
    symptoms = Column(JSON, nullable=True)
    
    # Timestamps
    requested_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    # Relationship
    session = relationship("TriageSessionModel", back_populates="bookings")


class NotificationModel(Base):
    """Notifications table - stores notifications for users."""
    __tablename__ = "notifications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_email = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String, nullable=False)  # "booking_approved", "booking_rejected", "new_request", etc.
    related_session_id = Column(String, nullable=True)
    related_booking_id = Column(String, nullable=True)
    read = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # JSON field for additional data
    extra_data = Column(JSON, nullable=True)


class MemoryModel(Base):
    """Memory/preferences table - stores user preferences."""
    __tablename__ = "memory"
    
    id = Column(String, primary_key=True)
    user_email = Column(String, unique=True, index=True, nullable=True)  # null = global/default
    preferred_city = Column(String, nullable=True)
    last_facility_used = Column(String, nullable=True)
    health_conditions = Column(JSON, nullable=True)  # Array of strings
    theme = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Track if database is initialized (for thread safety)
_db_initialized = False
import threading
_init_lock = threading.Lock()

# Create all tables
def init_db():
    """Initialize database - create all tables."""
    global _db_initialized
    
    # Double-check locking pattern for thread safety
    if _db_initialized:
        return
    
    with _init_lock:
        if _db_initialized:
            return
        
        try:
            # Ensure /tmp directory exists (for SQLite on Vercel only)
            if "sqlite" in DATABASE_URL and os.getenv("VERCEL"):
                os.makedirs("/tmp", exist_ok=True)
            
            # Create all tables
            Base.metadata.create_all(bind=engine)
            _db_initialized = True
            
            import logging
            db_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
            logging.info(f"Database ({db_type}) initialized successfully")
            
            # Test connection
            try:
                with engine.connect() as conn:
                    from sqlalchemy import text
                    conn.execute(text("SELECT 1"))
                logging.info("Database connection test successful")
            except Exception as conn_error:
                import logging
                logging.error(f"Database connection test failed: {conn_error}", exc_info=True)
                # On Vercel with PostgreSQL, this might indicate missing env vars
                if os.getenv("VERCEL") and "postgresql" in DATABASE_URL:
                    logging.error("PostgreSQL connection failed on Vercel. Check SUPABASE_URL and SUPABASE_DB_PASSWORD env vars.")
        except Exception as e:
            import logging
            logging.error(f"Database initialization error: {e}", exc_info=True)
            # Don't raise - mark as initialized to prevent retry loops
            # The error might be that tables already exist, which is fine
            _db_initialized = True


# Dependency to get database session
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

