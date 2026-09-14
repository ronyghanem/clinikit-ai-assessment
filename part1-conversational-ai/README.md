# Part 1 — Conversational / Agentic AI

## Overview

This project implements a simple AI-powered medical clinic assistant.

The assistant receives a patient's message and:

1. Identifies the user's intent.
2. Extracts relevant appointment information.
3. Determines the appropriate next action.
4. Responds safely and clearly.
5. Avoids making appointment changes without confirmation.

The system is designed as a lightweight rule-based conversational agent that demonstrates safe decision-making in a medical-clinic scenario.

---

## Supported Intents

The assistant can identify the following intents:

- `book_appointment`
- `reschedule_appointment`
- `cancel_appointment`
- `opening_hours`
- `doctor_availability`
- `human_handoff`
- `unclear`

---

## Supported Actions

Depending on the detected intent and available information, the assistant can choose one of the following actions:

- `create_appointment()`
- `reschedule_appointment()`
- `cancel_appointment()`
- `check_availability()`
- `handoff_to_human()`
- `ask_for_more_information()`

The assistant does not directly modify appointments. Instead, it returns a structured decision indicating what action should be taken.

---

## Safety Rules

The assistant follows several safety rules:

### 1. Confirmation Before Booking

The assistant does not create an appointment immediately.

For example:

> Book an appointment with Dr. George on Monday at 10 AM.

The assistant asks the patient to confirm the appointment first.

### 2. No Action When the User Says Not to Confirm

If the patient says:

> Book me Friday at 4 PM, but don't confirm anything yet.

The assistant does not create the appointment.

### 3. Missing Information

If important information is missing, the assistant asks for clarification.

For example, if the patient does not provide a doctor name, the assistant requests the missing doctor information.

### 4. Ambiguous Requests

The assistant treats uncertain requests safely.

For example:

> I might want to see Dr. George tomorrow at 4, but don't book anything yet.

This is not treated as a confirmed booking request.

### 5. Human Handoff

Requests that require human support can be redirected to a clinic representative.

---

## Project Structure

```text
part1-conversational-ai/
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   └── ...
│
├── tests/
│   ├── test_agent.py
│   └── ...
│
├── requirements.txt
└── README.md
Installation

Create and activate a virtual environment.

Windows
python -m venv .venv
.venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt
Running the Assistant

From the part1-conversational-ai directory, run:

python -m src.agent

The program processes example patient messages and displays the extracted information and recommended action.

Running the Tests

Run the test suite with:

pytest -q

The current test suite contains 10 tests covering:

Appointment booking
Appointment confirmation
Missing appointment information
Rescheduling
Cancellation
Opening-hours requests
Doctor availability
Human handoff
Ambiguous requests
Safety behavior

Expected result:

10 passed
Example
Input
Book an appointment with Dr. George on Monday at 10 AM.
Output
Intent: book_appointment
Doctor: Dr. George
Date: Monday
Time: 10 AM
Action: create_appointment
Should execute: false
Requires confirmation: true

The assistant asks the patient for confirmation before creating the appointment.

Another Example
Input
Book me Friday at 4 PM, but don't confirm anything yet.
Output
Intent: book_appointment
Doctor: None
Date: Friday
Time: 4 PM
Action: ask_for_more_information
Should execute: false
Requires confirmation: false

The assistant does not create an appointment because the user explicitly said not to confirm anything yet.

Design Approach

The assistant uses a rule-based approach based on:

Keyword and phrase matching
Intent classification
Basic entity extraction
Appointment information validation
Safety checks
Structured action selection

The response includes structured fields such as:

Detected intent
Extracted doctor
Extracted date
Extracted time
Selected action
Whether the action should be executed
Whether confirmation is required
Missing information
Confidence score
User-facing response
Limitations

This project is a simplified educational prototype.

It does not include:

A real medical-clinic database
Real appointment scheduling APIs
Real doctor availability data
Authentication or authorization
Persistent conversation memory
Advanced natural-language understanding
Integration with a hospital or clinic management system

The assistant should not be used for real medical appointment management without additional validation, security controls, and integration testing.

Conclusion

This project demonstrates how a conversational clinic assistant can identify user intent, extract appointment details, choose an appropriate action, and apply safety rules before performing appointment-related operations.