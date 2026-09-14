import pytest

from src.agent import CliniKitAgent


@pytest.fixture
def agent():
    return CliniKitAgent()


def test_booking_requires_confirmation(agent):
    result = agent.process_message(
        "Book an appointment with Dr. George on Monday at 10 AM."
    )

    assert result.intent == "book_appointment"
    assert result.doctor == "Dr. George"
    assert result.preferred_date == "Monday"
    assert result.preferred_time == "10 AM"

    assert result.action in {
        "create_appointment",
        "ask_for_more_information",
    }

    assert result.should_execute is False


def test_do_not_confirm_prevents_booking(agent):
    result = agent.process_message(
        "Book me Friday at 4 but don't confirm anything yet."
    )

    assert result.intent == "book_appointment"
    assert result.should_execute is False
    assert result.action == "ask_for_more_information"


def test_availability_is_not_booking(agent):
    result = agent.process_message(
        "Can I check if Dr. Smith is available tomorrow morning?"
    )

    assert result.intent in {
        "doctor_availability",
        "book_appointment",
    }

    assert result.action == "check_availability"
    assert result.should_execute is False
    assert result.doctor == "Dr. Smith"


def test_human_handoff(agent):
    result = agent.process_message(
        "Can somebody from the clinic call me?"
    )

    assert result.intent == "human_handoff"
    assert result.action == "handoff_to_human"


def test_cancellation_is_safe(agent):
    result = agent.process_message(
        "I need to cancel my appointment with Dr. Ali tomorrow."
    )

    assert result.intent == "cancel_appointment"
    assert result.should_execute is False
    assert result.action == "ask_for_more_information"


def test_rescheduling_is_safe(agent):
    result = agent.process_message(
        "Please move my appointment from Monday to Wednesday at 3 PM."
    )

    assert result.intent == "reschedule_appointment"
    assert result.should_execute is False

    assert result.action in {
        "reschedule_appointment",
        "ask_for_more_information",
    }


def test_uncertain_booking_does_not_execute(agent):
    result = agent.process_message(
        "I want an appointment but I'm not sure about the date yet."
    )

    assert result.intent == "book_appointment"
    assert result.should_execute is False
    assert result.action == "ask_for_more_information"


def test_clinic_hours_do_not_get_invented(agent):
    result = agent.process_message(
        "What time does the clinic open?"
    )

    assert result.intent == "clinic_hours"
    assert result.should_execute is False
    assert result.action == "provide_information"


def test_empty_message_is_rejected(agent):
    with pytest.raises(ValueError):
        agent.process_message("")


def test_whitespace_message_is_rejected(agent):
    with pytest.raises(ValueError):
        agent.process_message("   ")