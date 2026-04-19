"""Multi-agent system: Intake, Triage, FacilityFinder, Coordinator."""
from typing import List, Optional
from app.models import (
    IntakeAnswers, TriageResult, Facility, Recommendation, BookingState
)
from app.tools import tool_geocode, tool_find_facilities, tool_eta_minutes, haversine_km
from app.llm_client import get_llm_client
from app.memory import load_memory, save_memory
from app.observability import METRICS, log_event
from app.observability import now_iso

llm = get_llm_client()


class TriageAgent:
    """Rule-based triage agent with optional LLM explanation."""
    def run(self, intake: IntakeAnswers) -> TriageResult:
        red_flags = [
            intake.unconscious,
            intake.breathing_difficulty,
            intake.chest_pain,
            intake.stroke_signs,
            intake.major_bleeding,
            intake.severe_allergy
        ]
        
        if any(red_flags):
            level = "emergency"
            reason = "One or more red-flag symptoms present (airway/breathing/circulation/neurologic risk)."
            action = "CALL EMERGENCY SERVICES (911/ambulance) NOW. Do not delay. Prefer ambulance/ER."
        elif intake.injury_trauma:
            level = "high"
            reason = "Trauma/injury reported without immediate red flags."
            action = "Seek urgent medical evaluation immediately. Consider ER/urgent care. If symptoms worsen, call emergency services."
        else:
            level = "medium" if len(intake.symptoms) >= 2 else "low"
            reason = "No immediate red flags reported."
            action = "If symptoms worsen or new red flags appear, escalate to emergency care."

        safety = (
            "⚠️ IMPORTANT: This tool does NOT diagnose. "
            "If you're unsure, treat it as urgent. "
            "For emergencies, call local emergency services (911/ambulance) immediately."
        )

        # Optional LLM explanation
        explanation = ""
        try:
            prompt = f"""You are a safety-focused emergency navigation assistant.
Return a short, calm explanation and next steps in plain language.
Do NOT diagnose. Always include a safety reminder.
Inputs:
- Symptoms: {intake.symptoms}
- Flags: unconscious={intake.unconscious}, breathing_difficulty={intake.breathing_difficulty}, chest_pain={intake.chest_pain}, stroke_signs={intake.stroke_signs}, major_bleeding={intake.major_bleeding}, severe_allergy={intake.severe_allergy}, trauma={intake.injury_trauma}
- Chosen Level: {level}
"""
            explanation = llm.generate(prompt).strip()
        except Exception as e:
            METRICS["errors"] += 1
            log_event("llm_error", where="TriageAgent", error=str(e))

        final_reason = reason + ("\n\nLLM Explanation:\n" + explanation if explanation else "")
        return TriageResult(level=level, reason=final_reason, recommended_action=action, safety_note=safety)


class FacilityFinderAgent:
    """Finds and ranks nearby facilities."""
    def run(self, origin_lat: float, origin_lon: float, triage_level: str) -> List[Facility]:
        if triage_level in ("emergency", "high"):
            kinds = [("hospital", "hospital"), ("clinic", "clinic")]
            search_terms = ["hospital", "emergency hospital", "ER", "clinic"]
        else:
            kinds = [("clinic", "clinic"), ("hospital", "hospital")]
            search_terms = ["clinic", "hospital"]

        candidates: List[Facility] = []
        for term in search_terms:
            for kind, klabel in kinds:
                try:
                    res = tool_find_facilities(origin_lat, origin_lon, term, klabel, limit=6)
                    candidates.extend(res)
                except Exception as e:
                    METRICS["errors"] += 1
                    log_event("tool_error", tool="find_facilities", error=str(e), term=term, kind=klabel)

        # De-dup
        uniq = {}
        for f in candidates:
            name_normalized = f.name.lower().strip().split(",")[0].strip()
            key = (name_normalized, round(f.lat, 3), round(f.lon, 3))
            if key not in uniq or (f.kind == "hospital" and uniq[key].kind != "hospital"):
                uniq[key] = f
        facilities = list(uniq.values())

        # Compute distances + ETA
        origin = (origin_lat, origin_lon)
        for f in facilities:
            f.distance_km = round(haversine_km(origin_lat, origin_lon, f.lat, f.lon), 2)
            eta = None
            try:
                eta = tool_eta_minutes(origin, (f.lat, f.lon))
            except Exception as e:
                METRICS["errors"] += 1
                log_event("tool_error", tool="eta", error=str(e))
            f.eta_minutes = eta

        # Rank: ETA first if available, else distance
        facilities.sort(key=lambda x: (
            x.eta_minutes if x.eta_minutes is not None else 9999,
            x.distance_km if x.distance_km is not None else 9999
        ))
        return facilities[:5]


class CoordinatorAgent:
    """Orchestrates the workflow."""
    def __init__(self):
        self.booking = BookingState()
        self.intake: Optional[IntakeAnswers] = None
        self.triage: Optional[TriageResult] = None
        self.recommendation: Optional[Recommendation] = None

    def build_handoff_packet(
        self, intake: IntakeAnswers, triage: TriageResult, chosen: Facility, eta: Optional[int]
    ) -> str:
        s = []
        s.append(f"S (Situation): Patient '{intake.name}' en route. Triage level: {triage.level}.")
        s.append(f"B (Background): Symptoms: {', '.join(intake.symptoms) if intake.symptoms else 'N/A'}.")
        flags = []
        for k in ["unconscious", "breathing_difficulty", "chest_pain", "stroke_signs", 
                  "major_bleeding", "severe_allergy", "injury_trauma", "pregnancy"]:
            if getattr(intake, k):
                flags.append(k)
        s.append(f"A (Assessment): Flags: {', '.join(flags) if flags else 'None reported'}.")
        s.append(f"R (Recommendation): Prepare triage on arrival. Estimated arrival: {eta if eta is not None else 'unknown'} min.")
        s.append("Note: This summary is generated for information support only; clinician judgement required.")
        s.append(f"Destination: {chosen.name} ({chosen.address})")
        return "\n".join(s)

    def run(self, intake: IntakeAnswers) -> Recommendation:
        log_event("coordinator_start", intake=intake.model_dump())
        self.intake = intake  # Store intake data for later retrieval

        # Geocode
        try:
            lat, lon, disp = tool_geocode(intake.location_query)
        except Exception as e:
            METRICS["errors"] += 1
            log_event("tool_error", tool="geocode", error=str(e))
            raise

        # Triage
        triage = TriageAgent().run(intake)
        self.triage = triage  # Store triage result

        # Find facilities
        facilities = FacilityFinderAgent().run(lat, lon, triage.level)
        if not facilities:
            raise RuntimeError("No facilities found. Try a broader location query.")

        chosen = facilities[0]
        eta = chosen.eta_minutes

        # Route notes
        route_notes = (
            f"Nearest recommended: {chosen.name}. Estimated travel time: {eta if eta is not None else 'unknown'} min. "
            "If traffic is heavy or symptoms worsen, consider calling emergency services."
        )

        # Handoff packet
        handoff = self.build_handoff_packet(intake, triage, chosen, eta)

        # Create hospital request based on triage level
        # Two modes: Emergency/High -> Pre-Arrival Alert, Medium/Low -> Appointment Booking
        self.booking.facility_name = chosen.name
        self.booking.requested_at = now_iso()
        self.booking.patient_name = intake.name
        self.booking.triage_level = triage.level
        
        if triage.level in ("emergency", "high"):
            # Emergency Mode: Pre-Arrival Alert
            self.booking.request_type = "alert"
            self.booking.status = "PENDING_ACK"
            self.booking.note = "Pre-arrival alert created. Hospital must acknowledge to confirm readiness."
            booking_status = "PENDING_ACK"
        else:
            # Non-Emergency Mode: Appointment Booking Request
            self.booking.request_type = "appointment"
            self.booking.status = "PENDING_APPROVAL"
            self.booking.note = "Appointment booking request created. Hospital must approve to confirm appointment."
            booking_status = "PENDING_APPROVAL"

        # Update memory
        mem = load_memory()
        mem.preferred_city = intake.location_query
        mem.last_facility_used = chosen.name
        save_memory(mem)

        log_event("coordinator_done", chosen=chosen.model_dump(), booking=self.booking.model_dump(), metrics=METRICS)

        recommendation = Recommendation(
            triage=triage,
            top_choices=facilities,
            route_notes=route_notes,
            handoff_packet=handoff,
            booking_status=booking_status,
            booking=self.booking
        )
        self.recommendation = recommendation  # Store recommendation
        return recommendation

    def ack_alert(self) -> BookingState:
        """
        Acknowledge a pre-arrival alert (Emergency/High cases).
        Sets status to ACKNOWLEDGED.
        """
        if self.booking.request_type != "alert":
            raise ValueError(f"Cannot acknowledge: request type is '{self.booking.request_type}', expected 'alert'")
        if self.booking.status != "PENDING_ACK":
            raise ValueError(f"Cannot acknowledge: status is '{self.booking.status}', expected 'PENDING_ACK'")
        
        self.booking.status = "ACKNOWLEDGED"
        self.booking.acknowledged_at = now_iso()
        self.booking.note = "Pre-arrival alert acknowledged. Hospital is ready to receive patient."
        log_event("alert_acknowledged", booking=self.booking.model_dump())
        return self.booking

    def approve_appointment(self) -> BookingState:
        """
        Approve an appointment booking request (Medium/Low cases).
        Sets status to CONFIRMED.
        """
        if self.booking.request_type != "appointment":
            raise ValueError(f"Cannot approve appointment: request type is '{self.booking.request_type}', expected 'appointment'")
        if self.booking.status != "PENDING_APPROVAL":
            raise ValueError(f"Cannot approve: status is '{self.booking.status}', expected 'PENDING_APPROVAL'")
        
        self.booking.status = "CONFIRMED"
        self.booking.approved_at = now_iso()
        self.booking.note = "Appointment booking confirmed."
        log_event("appointment_approved", booking=self.booking.model_dump())
        return self.booking

    def reject_request(self, reason: Optional[str] = None) -> BookingState:
        """
        Reject any request (alert or appointment).
        Sets status to REJECTED.
        """
        if self.booking.status not in ("PENDING_ACK", "PENDING_APPROVAL"):
            raise ValueError(f"Cannot reject: status is '{self.booking.status}', expected PENDING_ACK or PENDING_APPROVAL")
        
        self.booking.status = "REJECTED"
        self.booking.note = reason or f"Request rejected by hospital."
        log_event("request_rejected", booking=self.booking.model_dump(), reason=reason)
        return self.booking

    # Backward compatibility method (deprecated, use ack_alert or approve_appointment)
    def approve_booking(self) -> BookingState:
        """Deprecated: Use ack_alert() or approve_appointment() instead."""
        if self.booking.request_type == "alert":
            return self.ack_alert()
        elif self.booking.request_type == "appointment":
            return self.approve_appointment()
        else:
            raise ValueError(f"Unknown request type: {self.booking.request_type}")


