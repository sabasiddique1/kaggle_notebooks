"""Database service layer for CRUD operations."""
from sqlalchemy.orm import Session
from app.database import (
    TriageSessionModel, BookingModel, MemoryModel, UserModel, NotificationModel,
    SessionLocal
)
from app.models import IntakeAnswers, TriageResult, Recommendation, BookingState
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import uuid


def _parse_iso_datetime(iso_string: str) -> datetime:
    """Parse ISO format datetime string, handling 'Z' suffix for UTC."""
    if not iso_string:
        return datetime.utcnow()
    
    # Replace 'Z' with '+00:00' for UTC timezone
    if iso_string.endswith('Z'):
        iso_string = iso_string[:-1] + '+00:00'
    
    try:
        return datetime.fromisoformat(iso_string)
    except ValueError:
        # Fallback: try parsing without timezone
        try:
            return datetime.fromisoformat(iso_string.replace('+00:00', ''))
        except ValueError:
            # Last resort: return current time
            return datetime.utcnow()


def save_triage_session(
    session_id: str,
    intake: IntakeAnswers,
    triage: TriageResult,
    recommendation: Recommendation,
    booking: BookingState,
    patient_email: Optional[str] = None
) -> TriageSessionModel:
    """Save a complete triage session to database."""
    db = SessionLocal()
    try:
        # Get location from intake
        location_lat = None
        location_lon = None
        if hasattr(intake, 'location_query') and intake.location_query:
            # Location will be geocoded in the coordinator, but we store the query
            pass
        
        session_model = TriageSessionModel(
            session_id=session_id,
            patient_email=patient_email,
            patient_name=intake.name,
            intake_data=intake.model_dump(),
            triage_level=triage.level,
            triage_reason=triage.reason,
            triage_action=triage.recommended_action,
            triage_safety_note=triage.safety_note,
            location_query=intake.location_query,
            recommendation_data=recommendation.model_dump(),
            request_type=booking.request_type,
            booking_status=booking.status,
            facility_name=booking.facility_name
        )
        
        db.add(session_model)
        db.commit()
        db.refresh(session_model)
        return session_model
    finally:
        db.close()


def save_booking(
    session_id: str,
    booking_data: Dict[str, Any]
) -> BookingModel:
    """Save a booking/request to database."""
    db = SessionLocal()
    try:
        booking_model = BookingModel(
            id=booking_data.get("id"),
            session_id=session_id,
            patient_email=booking_data.get("patient_email"),
            patient_name=booking_data.get("patient_name"),
            facility_name=booking_data.get("facility_name"),
            triage_level=booking_data.get("triage_level"),
            request_type=booking_data.get("request_type"),
            status=booking_data.get("status"),
            handoff_packet=booking_data.get("handoff_packet"),
            eta_minutes=booking_data.get("eta_minutes"),
            location=booking_data.get("location"),
            symptoms=booking_data.get("symptoms", []),
            requested_at=_parse_iso_datetime(booking_data.get("requested_at")) if booking_data.get("requested_at") else datetime.utcnow()
        )
        
        db.add(booking_model)
        db.commit()
        db.refresh(booking_model)
        return booking_model
    finally:
        db.close()


def update_booking_status(
    session_id: str,
    status: str,
    approved_at: Optional[datetime] = None,
    acknowledged_at: Optional[datetime] = None,
    rejected_at: Optional[datetime] = None,
    rejection_reason: Optional[str] = None
) -> Optional[BookingModel]:
    """Update booking status."""
    db = SessionLocal()
    try:
        booking = db.query(BookingModel).filter(BookingModel.session_id == session_id).first()
        if not booking:
            return None
        
        booking.status = status
        if approved_at:
            booking.approved_at = approved_at
        if acknowledged_at:
            booking.acknowledged_at = acknowledged_at
        if rejected_at:
            booking.rejected_at = rejected_at
        if rejection_reason:
            booking.rejection_reason = rejection_reason
        
        db.commit()
        db.refresh(booking)
        return booking
    finally:
        db.close()


def get_booking_by_session_id(session_id: str) -> Optional[BookingModel]:
    """Get booking by session ID."""
    db = SessionLocal()
    try:
        return db.query(BookingModel).filter(BookingModel.session_id == session_id).first()
    finally:
        db.close()


def get_bookings_by_facility(facility_name: str, status: Optional[str] = None) -> List[BookingModel]:
    """Get bookings for a specific facility."""
    db = SessionLocal()
    try:
        query = db.query(BookingModel).filter(BookingModel.facility_name == facility_name)
        if status:
            query = query.filter(BookingModel.status == status)
        return query.order_by(BookingModel.requested_at.desc()).all()
    finally:
        db.close()


def get_patient_bookings(patient_email: str) -> List[BookingModel]:
    """Get all bookings for a patient."""
    db = SessionLocal()
    try:
        return db.query(BookingModel).filter(
            BookingModel.patient_email == patient_email
        ).order_by(BookingModel.requested_at.desc()).all()
    finally:
        db.close()


def get_pending_bookings_count(patient_email: str) -> int:
    """Get count of pending bookings for a patient."""
    db = SessionLocal()
    try:
        return db.query(BookingModel).filter(
            BookingModel.patient_email == patient_email,
            BookingModel.status.in_(["PENDING_ACK", "PENDING_APPROVAL", "pending_approval"])
        ).count()
    finally:
        db.close()


def get_requested_hospitals(patient_email: str) -> List[str]:
    """Get list of hospitals the patient has already requested."""
    db = SessionLocal()
    try:
        bookings = db.query(BookingModel).filter(
            BookingModel.patient_email == patient_email,
            BookingModel.status.in_(["PENDING_ACK", "PENDING_APPROVAL", "pending_approval"])
        ).all()
        return list(set([b.facility_name for b in bookings]))
    finally:
        db.close()


def get_session_by_id(session_id: str) -> Optional[TriageSessionModel]:
    """Get triage session by ID."""
    db = SessionLocal()
    try:
        return db.query(TriageSessionModel).filter(TriageSessionModel.session_id == session_id).first()
    finally:
        db.close()


def get_patient_sessions(patient_email: str) -> List[TriageSessionModel]:
    """Get all sessions for a patient."""
    db = SessionLocal()
    try:
        return db.query(TriageSessionModel).filter(
            TriageSessionModel.patient_email == patient_email
        ).order_by(TriageSessionModel.created_at.desc()).all()
    finally:
        db.close()


def get_all_facilities() -> List[str]:
    """Get all unique facility names."""
    db = SessionLocal()
    try:
        facilities = db.query(BookingModel.facility_name).distinct().all()
        return [f[0] for f in facilities if f[0]]
    finally:
        db.close()


def get_memory(user_email: Optional[str] = None) -> Optional[MemoryModel]:
    """Get memory/preferences for a user or global default."""
    db = SessionLocal()
    try:
        if user_email:
            memory = db.query(MemoryModel).filter(MemoryModel.user_email == user_email).first()
            if memory:
                return memory
        
        # Return global/default memory
        return db.query(MemoryModel).filter(MemoryModel.user_email.is_(None)).first()
    finally:
        db.close()


def save_memory(
    user_email: Optional[str],
    preferred_city: Optional[str] = None,
    last_facility_used: Optional[str] = None,
    health_conditions: Optional[List[str]] = None,
    theme: Optional[str] = None
) -> MemoryModel:
    """Save or update memory/preferences."""
    db = SessionLocal()
    try:
        memory = db.query(MemoryModel).filter(MemoryModel.user_email == user_email).first()
        
        if not memory:
            memory = MemoryModel(
                id=str(uuid.uuid4()),
                user_email=user_email
            )
            db.add(memory)
        
        if preferred_city is not None:
            memory.preferred_city = preferred_city
        if last_facility_used is not None:
            memory.last_facility_used = last_facility_used
        if health_conditions is not None:
            memory.health_conditions = health_conditions
        if theme is not None:
            memory.theme = theme
        
        db.commit()
        db.refresh(memory)
        return memory
    finally:
        db.close()


# Helper to convert database models to API response format
def booking_to_dict(booking: BookingModel) -> Dict[str, Any]:
    """Convert BookingModel to dictionary for API response."""
    return {
        "id": booking.id,
        "session_id": booking.session_id,
        "facility_name": booking.facility_name,
        "patient_name": booking.patient_name,
        "triage_level": booking.triage_level,
        "request_type": booking.request_type,
        "status": booking.status,
        "requested_at": booking.requested_at.isoformat() if booking.requested_at else None,
        "approved_at": booking.approved_at.isoformat() if booking.approved_at else None,
        "acknowledged_at": booking.acknowledged_at.isoformat() if booking.acknowledged_at else None,
        "rejected_at": booking.rejected_at.isoformat() if booking.rejected_at else None,
        "rejection_reason": booking.rejection_reason,
        "handoff_packet": booking.handoff_packet,
        "eta_minutes": booking.eta_minutes,
        "symptoms": booking.symptoms or [],
        "location": booking.location
    }


# Notification functions
def create_notification(
    user_email: str,
    title: str,
    message: str,
    notification_type: str,
    related_session_id: Optional[str] = None,
    related_booking_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> NotificationModel:
    """Create a new notification."""
    db = SessionLocal()
    try:
        notification = NotificationModel(
            id=str(uuid.uuid4()),
            user_email=user_email,
            title=title,
            message=message,
            type=notification_type,
            related_session_id=related_session_id,
            related_booking_id=related_booking_id,
            metadata=metadata,
            read=False
        )
        db.add(notification)
        db.commit()
        db.refresh(notification)
        return notification
    finally:
        db.close()


def get_notifications(user_email: str, unread_only: bool = False, limit: Optional[int] = None) -> List[NotificationModel]:
    """Get notifications for a user."""
    db = SessionLocal()
    try:
        query = db.query(NotificationModel).filter(NotificationModel.user_email == user_email)
        if unread_only:
            query = query.filter(NotificationModel.read == False)
        query = query.order_by(NotificationModel.created_at.desc())
        if limit:
            query = query.limit(limit)
        return query.all()
    finally:
        db.close()


def get_unread_count(user_email: str) -> int:
    """Get count of unread notifications for a user."""
    db = SessionLocal()
    try:
        return db.query(NotificationModel).filter(
            NotificationModel.user_email == user_email,
            NotificationModel.read == False
        ).count()
    finally:
        db.close()


def mark_notification_read(notification_id: str) -> Optional[NotificationModel]:
    """Mark a notification as read."""
    db = SessionLocal()
    try:
        notification = db.query(NotificationModel).filter(NotificationModel.id == notification_id).first()
        if notification:
            notification.read = True
            db.commit()
            db.refresh(notification)
        return notification
    finally:
        db.close()


def mark_all_notifications_read(user_email: str) -> int:
    """Mark all notifications as read for a user."""
    db = SessionLocal()
    try:
        count = db.query(NotificationModel).filter(
            NotificationModel.user_email == user_email,
            NotificationModel.read == False
        ).update({"read": True})
        db.commit()
        return count
    finally:
        db.close()


def get_hospital_staff_emails(facility_name: str) -> List[str]:
    """Get all hospital staff emails for a facility."""
    db = SessionLocal()
    try:
        staff = db.query(UserModel).filter(
            UserModel.role == "hospital_staff",
            UserModel.facility_name == facility_name
        ).all()
        return [user.email for user in staff]
    finally:
        db.close()

