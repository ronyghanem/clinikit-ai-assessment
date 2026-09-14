from typing import Optional


def check_availability(
    doctor: Optional[str],
    date: Optional[str],
    time: Optional[str],
) -> str:
    return (
        f"Availability checked for "
        f"{doctor or 'any doctor'} on "
        f"{date or 'the requested date'} at "
        f"{time or 'the requested time'}."
    )


def create_appointment(
    doctor: Optional[str],
    date: Optional[str],
    time: Optional[str],
) -> str:
    return (
        f"Appointment created with "
        f"{doctor or 'the requested doctor'} on "
        f"{date or 'the requested date'} at "
        f"{time or 'the requested time'}."
    )


def reschedule_appointment(
    doctor: Optional[str],
    current_date: Optional[str],
    new_date: Optional[str],
    new_time: Optional[str],
) -> str:
    return (
        f"Appointment rescheduled from "
        f"{current_date or 'the current date'} to "
        f"{new_date or 'the requested date'} "
        f"at {new_time or 'the current time'}."
    )


def cancel_appointment(
    doctor: Optional[str],
    date: Optional[str],
) -> str:
    return (
        f"Appointment with {doctor or 'the requested doctor'} "
        f"on {date or 'the requested date'} has been cancelled."
    )


def handoff_to_human() -> str:
    return "A clinic representative will assist you shortly."