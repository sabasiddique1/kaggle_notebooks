"""FastAPI server for EmergencyCareNavigator."""
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Dict, Any, List, Optional
import uuid
import os
import base64
import logging
import traceback

from app.models import IntakeAnswers, Recommendation, BookingState, HospitalBooking, LoginRequest, RegisterRequest, AuthResponse, User
from datetime import datetime, timedelta
from app.agents import CoordinatorAgent
from app.observability import log_event
from app.llm_client import get_llm_client
from app.auth import (
    authenticate_user, create_user, create_access_token, verify_token,
    get_user_by_email, ACCESS_TOKEN_EXPIRE_MINUTES
)
from fastapi import Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.database import init_db
from app.db_service import (
    save_triage_session, save_booking, update_booking_status,
    get_booking_by_session_id, get_bookings_by_facility, get_patient_bookings,
    get_pending_bookings_count, get_requested_hospitals, get_session_by_id,
    get_patient_sessions, booking_to_dict, get_memory, save_memory,
    _parse_iso_datetime, create_notification, get_notifications, get_unread_count,
    mark_notification_read, mark_all_notifications_read, get_hospital_staff_emails
)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    app = FastAPI(title="EmergencyCareNavigator API")
    logger.info("FastAPI app created successfully")
except Exception as e:
    logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
    raise

# Global exception handler to ensure JSON responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return JSON."""
    import traceback
    error_trace = traceback.format_exc()
    logging.error(f"Unhandled exception at {request.url}: {exc}\n{error_trace}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors and return JSON."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup."""
    try:
        init_db()
        log_event("database_initialized", message="Database initialized successfully")
        
        # Ensure demo users exist in database (for Vercel cold starts)
        from app.auth import init_demo_users
        from app.database import SessionLocal, UserModel
        
        db = SessionLocal()
        try:
            user_count = db.query(UserModel).count()
            if user_count == 0:
                logging.info("No users found in database, initializing demo users...")
                try:
                    init_demo_users()
                    logging.info("Demo users initialized successfully")
                except Exception as e:
                    logging.warning(f"Failed to initialize demo users on startup: {e}")
            else:
                logging.info(f"Found {user_count} existing users in database")
        except Exception as e:
            logging.error(f"Error checking users: {e}")
        finally:
            db.close()
    except Exception as e:
        logging.error(f"Startup error: {e}", exc_info=True)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store coordinator instances per session
sessions: Dict[str, CoordinatorAgent] = {}

# Store all bookings for hospital panel access (keyed by facility name)
# Format: {facility_name: [booking1, booking2, ...]}
hospital_bookings: Dict[str, List[Dict[str, Any]]] = {}

# Store all patient requests (keyed by patient email or session_id for persistence)
# Format: {patient_identifier: [request1, request2, ...]}
patient_requests: Dict[str, List[Dict[str, Any]]] = {}

# Store uploaded documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Authentication
security = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    email = payload.get("sub")
    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = get_user_by_email(email)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get current authenticated user from JWT token (optional - returns None if not authenticated)."""
    if not credentials:
        return None
    try:
        token = credentials.credentials
        if not token:
            return None
            
        payload = verify_token(token)
        if payload is None:
            return None
        email = payload.get("sub")
        if email is None:
            return None
        user = get_user_by_email(email)
        return user
    except Exception:
        return None


def require_role(required_role: str):
    """Dependency to require specific role."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(status_code=403, detail=f"Access denied. Required role: {required_role}")
        return current_user
    return role_checker


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("static/index.html")


@app.post("/api/intake")
async def process_intake(intake: IntakeAnswers, patient_email: Optional[str] = Query(None)):
    """Process emergency intake and return recommendations."""
    session_id = str(uuid.uuid4())
    # Define patient_key at the very beginning to ensure it's always available
    patient_key = patient_email or session_id
    
    coordinator = CoordinatorAgent()
    sessions[session_id] = coordinator  # Keep in memory for active sessions
    
    try:
        result = coordinator.run(intake)
        result_dict = result.model_dump()
        result_dict["session_id"] = session_id  # Include session ID for booking approval
        result_dict["booking"] = coordinator.booking.model_dump()
        
        # Save session to database
        save_triage_session(
            session_id=session_id,
            intake=intake,
            triage=result.triage,
            recommendation=result,
            booking=coordinator.booking,
            patient_email=patient_email
        )
        
        # Store request in database (both alerts and appointments)
        
        if coordinator.booking.facility_name and coordinator.booking.status in ("PENDING_ACK", "PENDING_APPROVAL"):
            # Check for 3 pending requests limit using database
            if patient_email:
                pending_count = get_pending_bookings_count(patient_email)
                if pending_count >= 3:
                    raise HTTPException(
                        status_code=400, 
                        detail="You have reached the limit of 3 pending requests. Please wait for responses or cancel existing requests."
                    )
                
                # Check if already requested this hospital
                requested_hospitals = get_requested_hospitals(patient_email)
                if coordinator.booking.facility_name in requested_hospitals:
                    raise HTTPException(
                        status_code=400,
                        detail=f"You already have a pending request for {coordinator.booking.facility_name}. Please wait for a response."
                    )
            
            booking_data = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "patient_email": patient_email,
                "facility_name": coordinator.booking.facility_name,
                "patient_name": intake.name,
                "triage_level": result.triage.level,
                "request_type": coordinator.booking.request_type,  # "alert" or "appointment"
                "status": coordinator.booking.status,  # PENDING_ACK or PENDING_APPROVAL
                "requested_at": coordinator.booking.requested_at or datetime.now().isoformat(),
                "approved_at": None,
                "acknowledged_at": None,
                "handoff_packet": result.handoff_packet,
                "eta_minutes": result.top_choices[0].eta_minutes if result.top_choices else None,
                "symptoms": intake.symptoms,
                "location": intake.location_query,
                "rejected_at": None,
                "rejection_reason": None
            }
            
            # Save booking to database
            save_booking(session_id, booking_data)
            
            facility_name = coordinator.booking.facility_name
            if facility_name not in hospital_bookings:
                hospital_bookings[facility_name] = []
            hospital_bookings[facility_name].append(booking_data)
            
            # Also store in patient_requests for persistence (use patient_email or session_id as key)
            if patient_key not in patient_requests:
                patient_requests[patient_key] = []
            patient_requests[patient_key].append(booking_data)
            
            # Create notification for hospital staff
            if facility_name:
                staff_emails = get_hospital_staff_emails(facility_name)
                request_type_text = "pre-arrival alert" if coordinator.booking.request_type == "alert" else "appointment request"
                for staff_email in staff_emails:
                    create_notification(
                        user_email=staff_email,
                        title=f"New {request_type_text.title()}",
                        message=f"{intake.name or 'Anonymous'} has submitted a {request_type_text} ({result.triage.level} priority). Action required.",
                        notification_type="new_booking_request",
                        related_session_id=session_id,
                        related_booking_id=booking_data.get("id"),
                        metadata={
                            "facility_name": facility_name,
                            "patient_name": intake.name,
                            "triage_level": result.triage.level,
                            "request_type": coordinator.booking.request_type
                        }
                    )
        
        return result_dict
    except HTTPException:
        raise
    except Exception as e:
        log_event("api_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/booking/ack-alert/{session_id}")
async def ack_alert(session_id: str) -> BookingState:
    """Acknowledge a pre-arrival alert (Emergency/High cases)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    coordinator = sessions[session_id]
    try:
        booking = coordinator.ack_alert()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Update database
    update_booking_status(
        session_id=session_id,
        status=booking.status,
        acknowledged_at=_parse_iso_datetime(booking.acknowledged_at) if booking.acknowledged_at else datetime.utcnow()
    )
    
    # Get patient email from session
    session_model = get_session_by_id(session_id)
    patient_email = session_model.patient_email if session_model else None
    
    # Create notification for patient
    if patient_email:
        create_notification(
            user_email=patient_email,
            title="Alert Acknowledged",
            message=f"Your pre-arrival alert to {booking.facility_name or 'the hospital'} has been acknowledged. The hospital is ready to receive you.",
            notification_type="alert_acknowledged",
            related_session_id=session_id,
            metadata={
                "facility_name": booking.facility_name,
                "status": booking.status
            }
        )
    
    # Keep in-memory update for backward compatibility
    if booking.facility_name and booking.facility_name in hospital_bookings:
        for b in hospital_bookings[booking.facility_name]:
            if b["session_id"] == session_id:
                b["status"] = "ACKNOWLEDGED"
                b["acknowledged_at"] = booking.acknowledged_at or datetime.now().isoformat()
                break
    
    # Update patient_requests
    for patient_key, requests in patient_requests.items():
        for r in requests:
            if r["session_id"] == session_id:
                r["status"] = "ACKNOWLEDGED"
                r["acknowledged_at"] = booking.acknowledged_at or datetime.now().isoformat()
                break
    
    return booking


@app.post("/api/booking/approve-appointment/{session_id}")
async def approve_appointment(session_id: str) -> BookingState:
    """Approve an appointment booking request (Medium/Low cases)."""
    # Try to get session from memory first
    if session_id not in sessions:
        # Try to load from database
        session_model = get_session_by_id(session_id)
        if not session_model:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        # Reconstruct coordinator from database
        coordinator = CoordinatorAgent()
        # Restore booking state from database
        if session_model.booking_status:
            coordinator.booking.status = session_model.booking_status
        if session_model.request_type:
            coordinator.booking.request_type = session_model.request_type
        if session_model.facility_name:
            coordinator.booking.facility_name = session_model.facility_name
        if session_model.triage_level:
            coordinator.booking.triage_level = session_model.triage_level
        
        # Store in memory for future operations
        sessions[session_id] = coordinator
    else:
        coordinator = sessions[session_id]
    
    try:
        booking = coordinator.approve_appointment()
    except ValueError as e:
        # Provide more detailed error message
        error_msg = str(e)
        if "request type" in error_msg.lower():
            error_msg += f" Current booking: request_type={coordinator.booking.request_type}, status={coordinator.booking.status}"
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Update database
    approved_at_dt = _parse_iso_datetime(booking.approved_at) if booking.approved_at else datetime.utcnow()
    update_booking_status(
        session_id=session_id,
        status=booking.status,
        approved_at=approved_at_dt
    )
    
    # Get patient email from session
    session_model = get_session_by_id(session_id)
    patient_email = session_model.patient_email if session_model else None
    
    # Create notification for patient
    if patient_email:
        create_notification(
            user_email=patient_email,
            title="Request Approved",
            message=f"Your {booking.request_type} request to {booking.facility_name or 'the hospital'} has been approved.",
            notification_type="booking_approved",
            related_session_id=session_id,
            metadata={
                "facility_name": booking.facility_name,
                "request_type": booking.request_type,
                "status": booking.status
            }
        )
    
    # Update hospital_bookings and patient_requests (in-memory for backward compatibility)
    if booking.facility_name and booking.facility_name in hospital_bookings:
        for b in hospital_bookings[booking.facility_name]:
            if b["session_id"] == session_id:
                b["status"] = "CONFIRMED"
                b["approved_at"] = booking.approved_at or datetime.now().isoformat()
                break
    
    # Update patient_requests
    for patient_key, requests in patient_requests.items():
        for r in requests:
            if r["session_id"] == session_id:
                r["status"] = "CONFIRMED"
                r["approved_at"] = booking.approved_at or datetime.now().isoformat()
                break
    
    return booking


# Backward compatibility endpoint (deprecated)
@app.post("/api/booking/approve/{session_id}")
async def approve_booking(session_id: str) -> BookingState:
    """Deprecated: Use /api/booking/ack-alert or /api/booking/approve-appointment instead."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    coordinator = sessions[session_id]
    try:
        booking = coordinator.approve_booking()  # This will route to ack_alert or approve_appointment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Update hospital_bookings
    if booking.facility_name and booking.facility_name in hospital_bookings:
        for b in hospital_bookings[booking.facility_name]:
            if b["session_id"] == session_id:
                b["status"] = booking.status
                if booking.acknowledged_at:
                    b["acknowledged_at"] = booking.acknowledged_at
                if booking.approved_at:
                    b["approved_at"] = booking.approved_at
                break
    
    return booking


@app.get("/api/booking/status/{session_id}")
async def get_booking_status(session_id: str) -> Dict[str, Any]:
    """Get current booking status for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    coordinator = sessions[session_id]
    booking = coordinator.booking
    
    return {
        "session_id": session_id,
        "booking": booking.model_dump(),
        "status": booking.status
    }


@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str) -> Dict[str, Any]:
    """Get full session details including intake, triage, and recommendation."""
    # Try to get from database first
    session_model = get_session_by_id(session_id)
    if session_model:
        return {
            "session_id": session_id,
            "intake": session_model.intake_data,
            "triage": {
                "level": session_model.triage_level,
                "reason": session_model.triage_reason,
                "recommended_action": session_model.triage_action,
                "safety_note": session_model.triage_safety_note
            },
            "recommendation": session_model.recommendation_data,
            "booking": {
                "request_type": session_model.request_type,
                "status": session_model.booking_status,
                "facility_name": session_model.facility_name
            }
        }
    
    # Fallback to in-memory sessions (for backward compatibility)
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    coordinator = sessions[session_id]
    
    return {
        "session_id": session_id,
        "intake": coordinator.intake.model_dump() if coordinator.intake else None,
        "triage": coordinator.triage.model_dump() if coordinator.triage else None,
        "recommendation": coordinator.recommendation.model_dump() if coordinator.recommendation else None,
        "booking": coordinator.booking.model_dump()
    }


@app.get("/api/sessions")
async def get_patient_sessions_endpoint(patient_email: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    """Get all sessions for a patient."""
    if not patient_email:
        return []
    
    # Get from database
    sessions_list = get_patient_sessions(patient_email)
    result = []
    for session in sessions_list:
        result.append({
            "session_id": session.session_id,
            "patient_name": session.patient_name or "Anonymous",
            "location": session.location_query or "Unknown",
            "triage_level": session.triage_level or "unknown",
            "facility_name": session.facility_name or "Unknown",
            "request_type": session.request_type or "unknown",
            "status": session.booking_status or "unknown",
            "requested_at": session.created_at.isoformat() if session.created_at else None,
            "booking_status": session.booking_status or "unknown"
        })
    
    return result


@app.post("/api/booking/reject/{session_id}")
async def reject_request(session_id: str, request_data: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Reject any request (alert or appointment)."""
    reason = request_data.get("reason") if request_data else None
    
    # Try to get session from memory first
    if session_id not in sessions:
        # Try to load from database
        session_model = get_session_by_id(session_id)
        if not session_model:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        # Reconstruct coordinator from database
        coordinator = CoordinatorAgent()
        # Restore booking state from database
        if session_model.booking_status:
            coordinator.booking.status = session_model.booking_status
        if session_model.request_type:
            coordinator.booking.request_type = session_model.request_type
        if session_model.facility_name:
            coordinator.booking.facility_name = session_model.facility_name
        if session_model.triage_level:
            coordinator.booking.triage_level = session_model.triage_level
        
        # Store in memory for future operations
        sessions[session_id] = coordinator
    else:
        coordinator = sessions[session_id]
    try:
        booking = coordinator.reject_request(reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Update database
    rejected_at_dt = datetime.utcnow()
    update_booking_status(
        session_id=session_id,
        status=booking.status,
        rejected_at=rejected_at_dt,
        rejection_reason=reason or "Request rejected by hospital"
    )
    
    # Get patient email from session
    session_model = get_session_by_id(session_id)
    patient_email = session_model.patient_email if session_model else None
    
    # Create notification for patient
    if patient_email:
        create_notification(
            user_email=patient_email,
            title="Request Rejected",
            message=f"Your {coordinator.booking.request_type} request to {coordinator.booking.facility_name or 'the hospital'} has been rejected. {reason or 'Please try another facility.'}",
            notification_type="booking_rejected",
            related_session_id=session_id,
            metadata={
                "facility_name": coordinator.booking.facility_name,
                "request_type": coordinator.booking.request_type,
                "rejection_reason": reason
            }
        )
    
    # Update hospital_bookings and patient_requests (in-memory for backward compatibility)
    rejected_at = datetime.now().isoformat()
    booking_found = False
    for facility_name, bookings in hospital_bookings.items():
        for b in bookings:
            if b["session_id"] == session_id:
                b["status"] = "REJECTED"
                b["rejected_at"] = rejected_at
                b["rejection_reason"] = reason or "Request rejected by hospital"
                booking_found = True
                break
        if booking_found:
            break
    
    # Also update patient_requests
    for patient_key, requests in patient_requests.items():
        for r in requests:
            if r["session_id"] == session_id:
                r["status"] = "REJECTED"
                r["rejected_at"] = rejected_at
                r["rejection_reason"] = reason or "Request rejected by hospital"
                break
    
    return {"status": "REJECTED", "session_id": session_id, "reason": reason, "booking": booking.model_dump()}


@app.get("/api/patient/requests")
async def get_patient_requests(
    patient_email: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None)
) -> List[Dict[str, Any]]:
    """Get all requests for a patient."""
    # Try to get email from auth token if not provided as query parameter
    if not patient_email:
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.replace("Bearer ", "")
                payload = verify_token(token)
                if payload:
                    email = payload.get("sub")
                    if email:
                        user = get_user_by_email(email)
                        if user:
                            patient_email = user.email
            except Exception:
                pass  # Ignore auth errors
    
    if not patient_email:
        raise HTTPException(status_code=400, detail="patient_email query parameter is required or user must be authenticated")
    
    # Get from database - includes all statuses (PENDING, CONFIRMED, REJECTED, etc.)
    bookings = get_patient_bookings(patient_email)
    return [booking_to_dict(b) for b in bookings]


@app.get("/api/patient/requests/pending")
async def get_pending_requests_count(patient_email: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Get count of pending requests for a patient."""
    if not patient_email:
        return {"count": 0, "requests": [], "requested_hospitals": []}
    
    # Get from database
    bookings = get_patient_bookings(patient_email)
    pending = [b for b in bookings if b.status in ("PENDING_ACK", "PENDING_APPROVAL", "pending_approval")]
    
    return {
        "count": len(pending),
        "requests": [booking_to_dict(b) for b in pending],
        "requested_hospitals": list(set([b.facility_name for b in pending]))
    }


# Keep old endpoints for backward compatibility (will be deprecated)
@app.get("/api/hospital/bookings/{facility_name}")
async def get_hospital_bookings_legacy(facility_name: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all bookings for a specific hospital/facility (legacy, use protected endpoint)."""
    # Get from database
    bookings = get_bookings_by_facility(facility_name, status)
    return [booking_to_dict(b) for b in bookings]


@app.get("/api/hospital/facilities")
async def get_hospitals_with_bookings_legacy() -> List[Dict[str, Any]]:
    """Get list of all facilities that have bookings (legacy, use protected endpoint)."""
    # Get from database
    from sqlalchemy import func
    from sqlalchemy import case
    from app.database import BookingModel, SessionLocal
    
    db = SessionLocal()
    try:
        # Check if database has been initialized (table exists)
        try:
            facilities_query = db.query(
                BookingModel.facility_name,
                func.count(BookingModel.id).label('total'),
                func.sum(case(
                    (BookingModel.status.in_(["PENDING_ACK", "PENDING_APPROVAL"]), 1),
                    else_=0
                )).label('pending')
            ).group_by(BookingModel.facility_name).all()
            
            facilities = []
            for facility_name, total, pending in facilities_query:
                if facility_name:
                    facilities.append({
                        "name": facility_name,
                        "pending_bookings": int(pending or 0),
                        "total_bookings": int(total or 0)
                    })
            
            facilities.sort(key=lambda x: x["pending_bookings"], reverse=True)
            return facilities
        except Exception as e:
            # If database not initialized or no bookings yet, return empty list
            log_event("database_error", error=str(e), endpoint="hospital/facilities")
            return []
    finally:
        db.close()


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "EmergencyCareNavigator"}


@app.get("/api/debug/auth")
async def debug_auth():
    """Debug endpoint to check authentication setup."""
    try:
        from app.database import SessionLocal, UserModel, engine, DATABASE_URL
        from app.auth import SECRET_KEY
        import os
        
        # Check environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_password = os.getenv("SUPABASE_DB_PASSWORD")
        database_url_env = os.getenv("DATABASE_URL")
        
        debug_info = {
            "vercel_env": os.getenv("VERCEL", "not set"),
            "database_type": "PostgreSQL" if "postgresql" in str(engine.url) else "SQLite",
            "database_url_set": bool(database_url_env),
            "database_url": str(engine.url).replace(str(engine.url.password) if engine.url.password else "", "***") if hasattr(engine.url, 'password') else str(engine.url),
            "supabase_url_set": bool(supabase_url),
            "supabase_url": supabase_url if supabase_url else "not set",
            "supabase_password_set": bool(supabase_password),
            "jwt_secret_set": SECRET_KEY != "your-secret-key-change-in-production",
            "jwt_secret_length": len(SECRET_KEY) if SECRET_KEY else 0,
        }
        
        # Add setup recommendations
        if os.getenv("VERCEL"):
            if not database_url_env and not (supabase_url and supabase_password):
                debug_info["setup_required"] = "Set SUPABASE_URL and SUPABASE_DB_PASSWORD in Vercel environment variables"
            elif not database_url_env and (supabase_url and supabase_password):
                debug_info["setup_status"] = "Supabase env vars detected - connection string should be built automatically"
        
        # Try to query database
        try:
            db = SessionLocal()
            try:
                user_count = db.query(UserModel).count()
                debug_info["database_accessible"] = True
                debug_info["user_count"] = user_count
                
                # Try to get a user
                test_user = db.query(UserModel).first()
                if test_user:
                    debug_info["sample_user"] = {
                        "email": test_user.email,
                        "role": test_user.role
                    }
            except Exception as db_error:
                debug_info["database_accessible"] = False
                debug_info["database_error"] = str(db_error)
            finally:
                db.close()
        except Exception as e:
            debug_info["database_error"] = str(e)
            debug_info["database_accessible"] = False
        
        return debug_info
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Notification endpoints
@app.get("/api/notifications")
async def get_notifications_endpoint(
    unread_only: bool = Query(False),
    limit: Optional[int] = Query(None),
    authorization: Optional[str] = Header(None)
) -> List[Dict[str, Any]]:
    """Get notifications for the current user."""
    user_email = None
    
    # Extract user email from auth token
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        user_email = user.email
        except Exception:
            pass
    
    if not user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    notifications = get_notifications(user_email, unread_only=unread_only, limit=limit)
    return [
        {
            "id": n.id,
            "user_email": n.user_email,
            "title": n.title,
            "message": n.message,
            "type": n.type,
            "related_session_id": n.related_session_id,
            "related_booking_id": n.related_booking_id,
            "read": n.read,
            "created_at": n.created_at.isoformat() if n.created_at else None,
            "metadata": n.extra_data
        }
        for n in notifications
    ]


@app.get("/api/notifications/unread-count")
async def get_unread_count_endpoint(
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Get count of unread notifications for the current user."""
    user_email = None
    
    # Extract user email from auth token
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        user_email = user.email
        except Exception:
            pass
    
    if not user_email:
        return {"count": 0}
    
    count = get_unread_count(user_email)
    return {"count": count}


@app.post("/api/notifications/{notification_id}/read")
async def mark_notification_read_endpoint(
    notification_id: str,
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Mark a notification as read."""
    user_email = None
    
    # Extract user email from auth token
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        user_email = user.email
        except Exception:
            pass
    
    if not user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    notification = mark_notification_read(notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    if notification.user_email != user_email:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {"status": "success", "notification_id": notification_id}


@app.post("/api/notifications/read-all")
async def mark_all_notifications_read_endpoint(
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Mark all notifications as read for the current user."""
    user_email = None
    
    # Extract user email from auth token
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        user_email = user.email
        except Exception:
            pass
    
    if not user_email:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    count = mark_all_notifications_read(user_email)
    return {"status": "success", "marked_read": count}


@app.get("/api/memory")
async def get_memory_endpoint(authorization: Optional[str] = Header(None)):
    """Get memory bank (preferred city, last facility, health conditions)."""
    user_email = None
    
    # Try to extract user email from token if provided
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        user_email = user.email
        except Exception:
            pass  # Ignore auth errors, use default memory
    
    # Try database first
    memory_model = get_memory(user_email)
    if memory_model:
        return {
            "preferred_city": memory_model.preferred_city,
            "last_facility_used": memory_model.last_facility_used,
            "health_conditions": memory_model.health_conditions or [],
            "theme": memory_model.theme
        }
    
    # Fallback to JSON file for backward compatibility
    from app.memory import load_memory
    memory = load_memory()
    return memory.model_dump()


@app.post("/api/memory")
async def update_memory_endpoint(memory_data: dict, authorization: Optional[str] = Header(None)):
    """Update memory bank (preferred city, last facility, health conditions). No authentication required."""
    user_email = None
    
    # Try to extract user email from token if provided (optional)
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                email = payload.get("sub")
                if email:
                    user = get_user_by_email(email)
                    if user:
                        user_email = user.email
        except Exception:
            pass  # Ignore auth errors, use global memory
    
    memory_model = save_memory(
        user_email=user_email,  # None = global memory
        preferred_city=memory_data.get("preferred_city"),
        last_facility_used=memory_data.get("last_facility_used"),
        health_conditions=memory_data.get("health_conditions"),
        theme=memory_data.get("theme")
    )
    
    return {
        "status": "success",
        "memory": {
            "preferred_city": memory_model.preferred_city,
            "last_facility_used": memory_model.last_facility_used,
            "health_conditions": memory_model.health_conditions or [],
            "theme": memory_model.theme
        }
    }


@app.post("/api/memory/health-conditions")
async def add_health_condition(condition_data: dict):
    """Add a health condition to user history."""
    from app.memory import save_memory, load_memory
    from app.models import MemoryBank
    
    condition = condition_data.get("condition", "").strip()
    if not condition:
        raise HTTPException(status_code=400, detail="Condition cannot be empty")
    
    memory = load_memory()
    if condition not in memory.health_conditions:
        memory.health_conditions.append(condition)
        save_memory(memory)
    
    return {"status": "success", "memory": memory.model_dump()}


@app.delete("/api/memory/health-conditions/{condition}")
async def remove_health_condition(condition: str):
    """Remove a health condition from user history."""
    from app.memory import save_memory, load_memory
    from app.models import MemoryBank
    
    memory = load_memory()
    if condition in memory.health_conditions:
        memory.health_conditions.remove(condition)
        save_memory(memory)
    
    return {"status": "success", "memory": memory.model_dump()}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a medical document and extract health conditions using AI."""
    from app.memory import save_memory, load_memory
    from app.models import MemoryBank
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Read file content (for text files, PDFs, and images)
    file_ext = os.path.splitext(file.filename)[1].lower()
    text_content = ""
    extraction_error = None
    
    try:
        if file_ext in [".txt", ".md"]:
            # Plain text files
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
        elif file_ext == ".pdf":
            # PDF files - try to extract text
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    for page_num, page in enumerate(pdf_reader.pages[:5]):  # Limit to first 5 pages
                        try:
                            text_parts.append(page.extract_text())
                        except Exception as e:
                            log_event("pdf_page_error", page=page_num, error=str(e))
                    text_content = "\n".join(text_parts)
                    if not text_content.strip():
                        text_content = f"[PDF file: {file.filename}] - Could not extract text from PDF. The PDF may be image-based or encrypted."
            except ImportError:
                # PyPDF2 not installed - try alternative or use LLM with file info
                text_content = f"[PDF file: {file.filename}] - PDF parsing library not installed. Please install PyPDF2: pip install PyPDF2"
                extraction_error = "PDF library not installed"
            except Exception as e:
                text_content = f"[PDF file: {file.filename}] - Error reading PDF: {str(e)}"
                extraction_error = str(e)
        elif file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            # Image files - try OCR
            try:
                from PIL import Image
                import pytesseract
                img = Image.open(file_path)
                text_content = pytesseract.image_to_string(img)
                if not text_content.strip():
                    text_content = f"[Image file: {file.filename}] - Could not extract text from image. The image may not contain readable text."
            except ImportError as e:
                missing = "pytesseract" if "pytesseract" in str(e) else "PIL/Pillow"
                text_content = f"[Image file: {file.filename}] - OCR library not installed. Please install: pip install pytesseract pillow"
                extraction_error = f"OCR library ({missing}) not installed"
            except Exception as e:
                text_content = f"[Image file: {file.filename}] - Error reading image: {str(e)}"
                extraction_error = str(e)
        else:
            text_content = f"[Unsupported file type: {file_ext}] - Only .txt, .md, .pdf, and image files are supported."
            extraction_error = f"Unsupported file type: {file_ext}"
    except Exception as e:
        text_content = f"Error reading file: {str(e)}"
        extraction_error = str(e)
    
    # Use LLM to extract health conditions
    llm = get_llm_client()
    extracted_conditions = []
    
    # Try extraction if we have actual text content (not an error message)
    # Also try if we have PDF/image text even if it starts with "[" (might be partial extraction)
    should_extract = (
        text_content and 
        not extraction_error and
        (not text_content.startswith("[") or "Could not extract" not in text_content)
    )
    
    if should_extract:
        try:
            # Limit text to avoid token limits (keep first 3000 chars for better extraction)
            text_for_extraction = text_content[:3000] if len(text_content) > 3000 else text_content
            
            prompt = f"""You are a medical document analyzer. Extract health conditions, medical diagnoses, and chronic conditions from the following medical document text.

Document text:
{text_for_extraction}

Return ONLY a comma-separated list of health conditions/diagnoses found in the document. 
Examples: diabetes, hypertension, asthma, chronic pain, arthritis, high blood pressure, heart disease
If no conditions are found, return "None".

Important: Only extract actual medical conditions/diagnoses. Ignore normal values, dates, or non-medical information.

Conditions:"""
            
            response = llm.generate(prompt).strip()
            log_event("document_extraction", filename=file.filename, response_length=len(response), response_preview=response[:200])
            
            # Parse response - handle various formats
            if response and response.lower() != "none":
                # Check if MockLLM response (doesn't contain actual conditions)
                if "MOCK mode" in response or "no API key" in response.lower():
                    # Fallback: Extract from document text directly using keyword matching
                    log_event("using_fallback_extraction", filename=file.filename)
                    medical_keywords = [
                        "diabetes", "hypertension", "asthma", "arthritis", "heart disease",
                        "high blood pressure", "chronic pain", "depression", "anxiety",
                        "migraine", "epilepsy", "thyroid", "kidney disease", "liver disease",
                        "copd", "osteoporosis", "fibromyalgia", "lupus", "rheumatoid arthritis",
                        "cancer", "tumor", "stroke", "heart attack", "pneumonia", "bronchitis"
                    ]
                    found_conditions = []
                    text_lower = text_content.lower()
                    for keyword in medical_keywords:
                        if keyword in text_lower:
                            # Capitalize properly
                            condition_name = keyword.title() if len(keyword.split()) == 1 else keyword.title()
                            if condition_name not in found_conditions:
                                found_conditions.append(condition_name)
                    extracted_conditions = found_conditions
                else:
                    # Clean up response - remove common prefixes
                    response = response.replace("Conditions:", "").replace("Health conditions:", "").replace("Extracted conditions:", "").strip()
                    # Split by comma and clean
                    conditions = [c.strip() for c in response.split(",") if c.strip()]
                    # Filter out "None" and empty strings
                    extracted_conditions = [c for c in conditions if c.lower() != "none" and len(c) > 0]
        except Exception as e:
            log_event("llm_extraction_error", error=str(e), filename=file.filename)
            extraction_error = f"LLM extraction failed: {str(e)}"
            # Try fallback extraction even on error
            try:
                medical_keywords = [
                    "diabetes", "hypertension", "asthma", "arthritis", "heart disease",
                    "high blood pressure", "chronic pain", "depression", "anxiety"
                ]
                found_conditions = []
                text_lower = text_content.lower()
                for keyword in medical_keywords:
                    if keyword in text_lower:
                        found_conditions.append(keyword.title())
                if found_conditions:
                    extracted_conditions = found_conditions
                    log_event("fallback_extraction_success", conditions=found_conditions)
            except:
                pass
    elif extraction_error:
        log_event("document_extraction_skipped", reason=extraction_error, filename=file.filename)
    
    # Save extracted conditions to memory
    if extracted_conditions:
        memory = load_memory()
        for condition in extracted_conditions:
            condition_clean = condition.strip().lower()
            # Check if already exists (case-insensitive)
            if not any(c.lower() == condition_clean for c in memory.health_conditions):
                memory.health_conditions.append(condition.strip())
        save_memory(memory)
    
    return {
        "status": "success",
        "filename": file.filename,
        "extracted_conditions": extracted_conditions,
        "message": f"File uploaded. Extracted {len(extracted_conditions)} condition(s)." if extracted_conditions else "File uploaded. No conditions extracted."
    }


# Authentication endpoints
@app.post("/api/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register a new user."""
    try:
        user = create_user(
            email=request.email,
            password=request.password,
            name=request.name,
            role=request.role,
            facility_name=request.facility_name
        )
        access_token = create_access_token(data={"sub": user.email, "role": user.role})
        return AuthResponse(
            access_token=access_token,
            user={
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "facility_name": user.facility_name
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login and get access token."""
    try:
        # CRITICAL FIX FOR VERCEL: Ensure database is initialized BEFORE any queries
        # On Vercel cold starts, startup event might not run before first request
        # This MUST happen synchronously before any database operations
        try:
            from app.database import init_db
            init_db()  # Thread-safe, idempotent - safe to call multiple times
            logging.info("Database initialization check completed")
        except Exception as init_error:
            logging.error(f"Database initialization failed: {init_error}", exc_info=True)
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Database initialization failed: {str(init_error)}",
                    "error_type": type(init_error).__name__,
                    "traceback": traceback.format_exc()
                }
            )
        
        # Ensure demo users exist in database (for cold starts on Vercel)
        try:
            from app.auth import init_demo_users, SECRET_KEY
            from app.database import SessionLocal, UserModel
        except ImportError as import_error:
            logging.error(f"Failed to import required modules: {import_error}", exc_info=True)
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Failed to import required modules: {str(import_error)}",
                    "error_type": type(import_error).__name__,
                    "traceback": traceback.format_exc()
                }
            )
        
        db = None
        try:
            db = SessionLocal()
            # Test database connection by querying user count
            user_count = db.query(UserModel).count()
            logging.info(f"Database connection successful. User count: {user_count}")
            
            if user_count == 0:
                try:
                    logging.info("No users found in database, initializing demo users...")
                    init_demo_users()
                    logging.info("Demo users initialized successfully")
                except Exception as init_error:
                    logging.error(f"Failed to initialize demo users: {init_error}", exc_info=True)
                    # Don't fail login if demo users fail - user might be registering
        except Exception as db_error:
            logging.error(f"Database query error in login: {db_error}", exc_info=True)
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Database query error: {str(db_error)}",
                    "error_type": type(db_error).__name__,
                    "hint": "Database might not be initialized. Check /api/debug/auth",
                    "traceback": traceback.format_exc()
                }
            )
        finally:
            if db:
                try:
                    db.close()
                except Exception:
                    pass
        
        # Verify JWT_SECRET_KEY is set
        if SECRET_KEY == "your-secret-key-change-in-production":
            logging.warning("JWT_SECRET_KEY not set! Using default key.")
        
        # Authenticate user
        try:
            from app.auth import authenticate_user, create_access_token
            user = authenticate_user(request.email, request.password)
        except ImportError as import_error:
            logging.error(f"Failed to import auth functions: {import_error}", exc_info=True)
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Failed to import auth functions: {str(import_error)}",
                    "error_type": type(import_error).__name__,
                    "traceback": traceback.format_exc()
                }
            )
        except Exception as auth_error:
            logging.error(f"Authentication error: {auth_error}", exc_info=True)
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Authentication error: {str(auth_error)}",
                    "error_type": type(auth_error).__name__,
                    "traceback": traceback.format_exc()
                }
            )
        
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        
        # Create access token
        try:
            access_token = create_access_token(data={"sub": user.email, "role": user.role})
        except Exception as token_error:
            logging.error(f"Token creation error: {token_error}", exc_info=True)
            import traceback
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Token creation failed: {str(token_error)}",
                    "error_type": type(token_error).__name__,
                    "traceback": traceback.format_exc()
                }
            )
        
        return AuthResponse(
            access_token=access_token,
            user={
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
                "facility_name": user.facility_name
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        try:
            log_event("login_error", error=str(e), email=request.email if hasattr(request, 'email') else "unknown")
        except Exception:
            pass  # Don't fail if logging fails
        error_details = traceback.format_exc()
        logging.error(f"Login failed: {error_details}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Login failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": error_details
            }
        )


@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "role": current_user.role,
        "facility_name": current_user.facility_name
    }


# Protected hospital endpoints
@app.get("/api/hospital/bookings/{facility_name}")
async def get_hospital_bookings_protected(
    facility_name: str,
    status: Optional[str] = None,
    current_user: User = Depends(require_role("hospital_staff"))
) -> List[Dict[str, Any]]:
    """Get all bookings for a specific hospital/facility (protected)."""
    # Verify user has access to this facility
    if current_user.facility_name and current_user.facility_name != facility_name:
        raise HTTPException(status_code=403, detail="Access denied to this facility")
    
    # Get from database for real-time data
    bookings = get_bookings_by_facility(facility_name, status)
    return [booking_to_dict(b) for b in bookings]


@app.get("/api/hospital/facilities")
async def get_hospitals_with_bookings_protected(
    current_user: User = Depends(require_role("hospital_staff"))
) -> List[Dict[str, Any]]:
    """Get list of all facilities that have bookings (protected)."""
    # Get from database for real-time data
    from sqlalchemy import func, case
    from app.database import BookingModel, SessionLocal
    
    db = SessionLocal()
    try:
        # Build query
        query = db.query(
            BookingModel.facility_name,
            func.count(BookingModel.id).label('total'),
            func.sum(case(
                (BookingModel.status.in_(["PENDING_ACK", "PENDING_APPROVAL", "pending_approval"]), 1),
                else_=0
            )).label('pending')
        ).group_by(BookingModel.facility_name)
        
        # Filter by user's facility if they have one
        if current_user.facility_name:
            query = query.filter(BookingModel.facility_name == current_user.facility_name)
        
        facilities_query = query.all()
        
        facilities = []
        for facility_name, total, pending in facilities_query:
            if facility_name:
                facilities.append({
                    "name": facility_name,
                    "pending_bookings": int(pending or 0),
                    "total_bookings": int(total or 0)
                })
        
        facilities.sort(key=lambda x: x["pending_bookings"], reverse=True)
        return facilities
    except Exception as e:
        log_event("database_error", error=str(e), endpoint="hospital/facilities")
        # Fallback to empty list
        return []
    finally:
        db.close()


@app.post("/api/booking/approve/{session_id}")
async def approve_booking_protected(
    session_id: str,
    current_user: User = Depends(require_role("hospital_staff"))
) -> BookingState:
    """Approve pending booking for a session (protected)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    coordinator = sessions[session_id]
    booking = coordinator.approve_booking()
    
    # Verify user has access to this facility
    if current_user.facility_name and booking.facility_name != current_user.facility_name:
        raise HTTPException(status_code=403, detail="Access denied to this facility")
    
    # Update hospital_bookings
    if booking.facility_name and booking.facility_name in hospital_bookings:
        for b in hospital_bookings[booking.facility_name]:
            if b["session_id"] == session_id:
                b["status"] = "confirmed"
                b["approved_at"] = booking.approved_at or datetime.now().isoformat()
                break
    
    return booking


@app.post("/api/booking/reject/{session_id}")
async def reject_request_protected(
    session_id: str,
    request_data: Dict[str, Any] = Body(default={}),
    current_user: User = Depends(require_role("hospital_staff"))
) -> Dict[str, Any]:
    """Reject any request (alert or appointment) (protected)."""
    reason = request_data.get("reason") if request_data else None
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    coordinator = sessions[session_id]
    
    # Verify user has access to this facility
    if current_user.facility_name and coordinator.booking.facility_name != current_user.facility_name:
        raise HTTPException(status_code=403, detail="Access denied to this facility")
    
    try:
        booking = coordinator.reject_request(reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Update hospital_bookings
    booking_found = False
    for facility_name, bookings in hospital_bookings.items():
        for b in bookings:
            if b["session_id"] == session_id:
                b["status"] = "REJECTED"
                b["rejected_at"] = datetime.now().isoformat()
                b["rejection_reason"] = reason or "Request rejected by hospital"
                booking_found = True
                break
        if booking_found:
            break
    
    return {"status": "REJECTED", "session_id": session_id, "reason": reason, "booking": booking.model_dump()}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
