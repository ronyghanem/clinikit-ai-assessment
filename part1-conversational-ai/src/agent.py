import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .actions import (
    cancel_appointment,
    check_availability,
    create_appointment,
    handoff_to_human,
    reschedule_appointment,
)
from .models import PatientRequest
from .prompts import SYSTEM_PROMPT


load_dotenv()


class CliniKitAgent:
    """
    CliniKit Conversational AI Agent.

    Responsibilities:
    1. Understand patient messages using an LLM.
    2. Extract structured information.
    3. Identify the appropriate action.
    4. Apply safety rules.
    5. Execute mocked clinic actions only when permitted.
    """

    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")

        if not api_key or api_key == "YOUR_GROQ_KEY":
            raise ValueError(
                "LLM_API_KEY is missing. Add a valid key to your .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.groq.com/openai/v1",
        )

        self.model = model or "openai/gpt-oss-20b"

    def process_message(self, message: str) -> PatientRequest:
        """
        Process one patient message and return structured information.
        """

        if not isinstance(message, str) or not message.strip():
            raise ValueError("Patient message cannot be empty.")

        data = self._ask_llm(message)

        if not isinstance(data, dict):
            raise ValueError("The AI returned an invalid JSON object.")

        # Some models may forget the response field.
        if not data.get("response"):
            data["response"] = self._generate_fallback_response(data)

        try:
            result = PatientRequest.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"The AI returned invalid structured data: {exc}"
            ) from exc

        self._apply_safety_rules(result, message)

        if result.should_execute:
            result.response = self._execute_action(result)

        return result

    def _ask_llm(self, message: str) -> dict[str, Any]:
        """
        Ask the LLM to classify and structure the patient message.

        The request is retried once if the provider returns an empty response.
        """

        last_error: Exception | None = None

        for attempt in range(2):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                )

                if not completion.choices:
                    raise ValueError("The AI returned no choices.")

                content = completion.choices[0].message.content

                if not content or not content.strip():
                    raise ValueError("The AI returned an empty response.")

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"The AI returned invalid JSON: {content}"
                    ) from exc

                if not isinstance(parsed, dict):
                    raise ValueError("The AI response must be a JSON object.")

                return parsed

            except Exception as exc:
                last_error = exc

                if attempt == 0:
                    continue

        raise RuntimeError(
            f"Unable to process the message with the AI provider: {last_error}"
        ) from last_error

    @staticmethod
    def _generate_fallback_response(data: dict[str, Any]) -> str:
        """
        Generate a safe response if the LLM omits the response field.
        """

        action = data.get("action")
        missing = data.get("missing_information", [])

        if action == "ask_for_more_information":
            if missing:
                readable = ", ".join(
                    str(field).replace("_", " ")
                    for field in missing
                )
                return f"Could you please provide your {readable}?"

            return "Could you please provide a little more information?"

        if action == "check_availability":
            return (
                "I will check the requested availability "
                "and let you know."
            )

        if action == "handoff_to_human":
            return "I will connect you with a clinic representative."

        if action == "provide_information":
            return (
                "I do not have that clinic information available yet. "
                "A clinic representative can help you with it."
            )

        if action == "create_appointment":
            return (
                "I can help arrange the appointment once the required "
                "information and confirmation are provided."
            )

        if action == "reschedule_appointment":
            return (
                "I can help reschedule the appointment once the required "
                "information and confirmation are provided."
            )

        if action == "cancel_appointment":
            return (
                "I can help cancel the appointment once the required "
                "information and confirmation are provided."
            )

        return "Could you please provide more information about your request?"

    @staticmethod
    def _apply_safety_rules(
        result: PatientRequest,
        message: str,
    ) -> None:
        """
        Apply safety rules before executing any clinic action.
        """

        # Rule 1: The patient explicitly does not want an action.
        if CliniKitAgent._contains_no_action_language(message):
            result.should_execute = False
            result.requires_confirmation = False

            if result.action in {
                "create_appointment",
                "reschedule_appointment",
                "cancel_appointment",
            }:
                result.action = "ask_for_more_information"

            result.response = (
                "I will not make any changes to your appointment. "
                "Please let me know when you are ready to proceed."
            )
            return

        # Rule 2: Creating an appointment requires date and time.
        if result.action == "create_appointment":
            missing = []

            if not result.preferred_date:
                missing.append("preferred_date")

            if not result.preferred_time:
                missing.append("preferred_time")

            if missing:
                result.should_execute = False
                result.requires_confirmation = False
                result.action = "ask_for_more_information"
                result.missing_information = missing
                result.response = (
                    "Could you please provide your preferred "
                    "date and time for the appointment?"
                )
                return

        # Rule 3: Rescheduling requires a new date.
        if result.action == "reschedule_appointment":
            if not result.preferred_date:
                result.should_execute = False
                result.requires_confirmation = False
                result.action = "ask_for_more_information"
                result.missing_information = ["preferred_date"]
                result.response = (
                    "Could you please provide the new date "
                    "for your appointment?"
                )
                return

              # Rule 4: Cancellation requires patient-identifying information.
        if result.action == "cancel_appointment":
            has_patient_identifier = bool(
                result.patient_id or result.patient_name
            )

            if not has_patient_identifier:
                result.should_execute = False
                result.requires_confirmation = False
                result.action = "ask_for_more_information"
                result.missing_information = [
                    "patient_name_or_patient_id"
                ]
                result.response = (
                    "Could you please provide your patient ID or name "
                    "so we can locate the appointment?"
                )
                return

        # Rule 5: State-changing actions always require confirmation.
        if result.action in {
            "create_appointment",
            "reschedule_appointment",
            "cancel_appointment",
        }:
            if not result.requires_confirmation:
                result.should_execute = False

            # Never execute unless confirmation is explicitly present.
            if not result.requires_confirmation:
                result.response = (
                    "Please confirm that you want me to proceed "
                    "with this appointment change."
                )

        # Rule 6: Availability checks and information requests
        # must never execute a state-changing action.
        if result.action in {
            "check_availability",
            "provide_information",
            "ask_for_more_information",
            "handoff_to_human",
        }:
            result.should_execute = False

    @staticmethod
    def _contains_no_action_language(message: str) -> bool:
        """
        Detect language showing uncertainty or explicitly rejecting action.
        """

        text = message.lower().strip()

        phrases = [
            "don't book",
            "do not book",
            "dont book",
            "don't confirm",
            "do not confirm",
            "dont confirm",
            "just asking",
            "just checking",
            "not ready to book",
            "maybe",
            "i might",
            "i'm not sure",
            "im not sure",
            "do not make changes",
            "don't make changes",
            "do not change",
            "don't change",
        ]

        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _execute_action(result: PatientRequest) -> str:
        """
        Execute a mocked clinic action.

        These functions do not modify a real clinic database.
        """

        if result.action == "check_availability":
            return check_availability(
                result.doctor,
                result.preferred_date,
                result.preferred_time,
            )

        if result.action == "create_appointment":
            return create_appointment(
                result.doctor,
                result.preferred_date,
                result.preferred_time,
            )

        if result.action == "reschedule_appointment":
            return reschedule_appointment(
                result.doctor,
                result.current_appointment_date,
                result.preferred_date,
                result.preferred_time,
            )

        if result.action == "cancel_appointment":
            return cancel_appointment(
                result.doctor,
                result.current_appointment_date,
            )

        if result.action == "handoff_to_human":
            return handoff_to_human()

        return result.response


def main() -> None:
    """
    Run the CliniKit agent through the command line.
    """

    try:
        agent = CliniKitAgent()
    except Exception as error:
        print(f"Startup error: {error}")
        return

    print("=" * 50)
    print("CliniKit AI Agent")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        try:
            message = input("\nPatient: ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if message.lower().strip() == "exit":
            print("Goodbye!")
            break

        try:
            result = agent.process_message(message)

            print("\nStructured output:")
            print(result.model_dump_json(indent=2))

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()