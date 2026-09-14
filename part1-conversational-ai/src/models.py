from typing import Optional, Literal
from pydantic import BaseModel, Field


Intent = Literal[
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "clinic_hours",
    "doctor_availability",
    "human_handoff",
    "other",
]

Action = Literal[
    "check_availability",
    "create_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "handoff_to_human",
    "ask_for_more_information",
    "provide_information",
]


class PatientRequest(BaseModel):
    intent: Intent

    patient_name: Optional[str] = None
    patient_id: Optional[str] = None

    doctor: Optional[str] = None

    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None

    current_appointment_date: Optional[str] = None
    current_appointment_time: Optional[str] = None

    action: Action

    should_execute: bool = False
    requires_confirmation: bool = False

    missing_information: list[str] = Field(default_factory=list)

    confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )

    response: str