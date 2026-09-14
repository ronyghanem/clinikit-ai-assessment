# CliniKit AI Trainee Assessment

This repository contains my solution to the CliniKit AI Trainee Assessment.

The assessment consists of two practical exercises:

1. Conversational / Agentic AI
2. Machine Learning

The solutions focus on safe decision-making, structured outputs, data preprocessing, model evaluation, and practical AI/ML workflows.

---

## Project Structure

```text
clinikit-ai-assessment/
│
├── part1-conversational-ai/
│   ├── src/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── part2-machine-learning/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── generate_data.py
│   ├── train.py
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
Part 1 — Conversational / Agentic AI
Overview

Part 1 implements a conversational medical clinic assistant.

The assistant receives a patient's message and:

Identifies the user's intent
Extracts relevant appointment information
Determines the appropriate next action
Requests missing information when necessary
Requires confirmation before making appointment changes
Handles ambiguous requests safely
Supported Intents
Booking an appointment
Rescheduling an appointment
Cancelling an appointment
Opening hours
Doctor availability
Human handoff
Unclear requests
Safety

The assistant does not directly create, reschedule, or cancel appointments without the appropriate confirmation.

For example:

Book an appointment with Dr. George on Monday at 10 AM.

The assistant requests confirmation before proceeding.

For ambiguous requests such as:

I might want to see Dr. George tomorrow at 4, but don't book anything yet.

the assistant does not execute the appointment action.

Testing

The Part 1 test suite contains 10 tests.

10 passed

To run:

cd part1-conversational-ai
.venv\Scripts\activate
pytest -q

To run the assistant:

python -m src.agent

More details are available in:

part1-conversational-ai/README.md

Part 2 — Machine Learning
Overview

Part 2 builds a machine learning system that predicts whether a patient will miss a scheduled appointment.

The workflow includes:

Dataset generation
Data exploration
Missing-value handling
Duplicate removal
Feature preprocessing
Model training
Model evaluation
Model selection
Model persistence
Example predictions
Dataset

No real patient dataset was provided with the assessment.

Therefore, a synthetic dataset containing 2,000 appointment records was generated specifically for this exercise.

Target distribution:

Attended: 1,553 (77.65%)
No-show: 447 (22.35%)

The dataset includes features such as:

Age
Gender
Appointment type
Days before appointment
Previous appointments
Previous no-shows
Weekday
Appointment time
Reminder status
New-patient status
Models

Two models were trained:

Logistic Regression
Random Forest
Results
Logistic Regression
Metric	Score
Accuracy	0.5600
Precision (No-show)	0.2723
Recall (No-show)	0.5843
F1-score (No-show)	0.3714
ROC-AUC	0.6165
Random Forest
Metric	Score
Accuracy	0.6600
Precision (No-show)	0.2920
Recall (No-show)	0.3708
F1-score (No-show)	0.3267
ROC-AUC	0.6213
Selected Model

Logistic Regression was selected because it achieved the higher F1-score for the no-show class:

Logistic Regression: 0.3714
Random Forest:       0.3267

The model and evaluation results are saved in the models/ and results/ directories.

Example Predictions

The prediction script produced the following example results:

0.145 → Likely to attend
0.873 → Likely to miss appointment
0.121 → Likely to attend
Running Part 2
cd part2-machine-learning
.venv\Scripts\activate
python generate_data.py
python train.py
python predict.py

More details are available in:

part2-machine-learning/README.md

Technologies
Part 1
Python
Pytest
Rule-based conversational logic
Structured JSON-style outputs
Part 2
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Joblib
Testing

Both parts were tested locally.

Part 1
10 passed
Part 2

The complete workflow was successfully executed:

python generate_data.py
python train.py
python predict.py

The trained model, metrics, feature importance, and confusion matrix were successfully generated.

Important Note

This repository is an educational assessment project.

The machine learning dataset is synthetic because no real patient dataset was provided.

The machine learning results should not be interpreted as real-world clinical performance.

The conversational assistant is also a prototype and is not intended for direct use in a real medical environment without additional validation, security, privacy controls, and integration testing.

Author

Rony Ghanem

Master's in Management Information Systems

GitHub: github.com/ronyghanem

LinkedIn: linkedin.com/in/rony-ghanem

Portfolio: ronygh.netlify.app