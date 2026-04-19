"""Data models for EmergencyCareNavigator."""
from typing import List, Optional
from pydantic import BaseModel, Field


class IntakeAnswers(BaseModel):
    name: str = Field(default="Anonymous")
    age_years: Optional[int] = None
    sex: Optional[str] = None  # 'M','F','Other'
    location_query: str = Field(..., description="User-provided location text")
    symptoms: List[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None

    # Red flags
    unconscious: bool = False
    breathing_difficulty: bool = False
    chest_pain: bool = False
    stroke_signs: bool = False
    major_bleeding: bool = False
    severe_allergy: bool = False
    pregnancy: bool = False
    injury_trauma: bool = False


class TriageResult(BaseModel):
    level: str  # "emergency"|"high"|"medium"|"low"
    reason: str
    recommended_action: str
    safety_note: str


class Facility(BaseModel):
    name: str
    address: str
    lat: float
    lon: float
    kind: str  # hospital/clinic/pharmacy
    distance_km: Optional[float] = None
    eta_minutes: Optional[int] = None
    source: str = "OSM"


class BookingState(BaseModel):
    """
    Represents a hospital request (alert or appointment booking).
    
    Two modes:
    - Emergency/High -> type="alert", status: PENDING_ACK -> ACKNOWLEDGED
    - Medium/Low -> type="appointment", status: PENDING_APPROVAL -> CONFIRMED
    """
    request_type: str = "alert"  # "alert" | "appointment"
    status: str = "not_started"  # For alerts: not_started | PENDING_ACK | ACKNOWLEDGED | REJECTED
                                 # For appointments: not_started | PENDING_APPROVAL | CONFIRMED | REJECTED
    facility_name: Optional[str] = None
    requested_at: Optional[str] = None
    approved_at: Optional[str] = None  # For appointments: when approved; for alerts: when acknowledged
    acknowledged_at: Optional[str] = None  # For alerts: when acknowledged
    note: Optional[str] = None
    session_id: Optional[str] = None
    patient_name: Optional[str] = None
    triage_level: Optional[str] = None


class Recommendation(BaseModel):
    triage: TriageResult
    top_choices: List[Facility]
    route_notes: str
    handoff_packet: str
    booking_status: str  # "not_started"|"PENDING_ACK"|"ACKNOWLEDGED"|"PENDING_APPROVAL"|"CONFIRMED"|"REJECTED"|"skipped"
    booking: Optional[BookingState] = None


class MemoryBank(BaseModel):
    preferred_city: Optional[str] = None
    last_facility_used: Optional[str] = None
    health_conditions: List[str] = Field(default_factory=list)


class HospitalBooking(BaseModel):
    id: str
    session_id: str
    facility_name: str
    patient_name: str
    triage_level: str
    request_type: str  # "alert" | "appointment"
    status: str  # PENDING_ACK | ACKNOWLEDGED | PENDING_APPROVAL | CONFIRMED | REJECTED
    requested_at: str
    approved_at: Optional[str] = None  # For appointments
    acknowledged_at: Optional[str] = None  # For alerts
    handoff_packet: Optional[str] = None
    eta_minutes: Optional[int] = None
    symptoms: Optional[List[str]] = None
    location: Optional[str] = None


class User(BaseModel):
    id: str
    email: str
    name: str
    role: str  # "patient" | "hospital_staff"
    facility_name: Optional[str] = None  # For hospital staff
    password_hash: Optional[str] = None  # Not sent to client


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str  # "patient" | "hospital_staff"
    facility_name: Optional[str] = None  # Required for hospital_staff


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict  # User info without password
