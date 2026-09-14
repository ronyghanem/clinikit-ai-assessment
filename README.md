# CliniKit AI Trainee Assessment

This repository contains my solution to the **CliniKit AI Trainee Assessment**.

The assessment consists of two practical exercises:

1. **Conversational / Agentic AI**
2. **Machine Learning**

The solutions focus on safe conversational decision-making, structured outputs, data preprocessing, model evaluation, and practical AI/ML workflows.

---

## Project Structure

```text
clinikit-ai-assessment/
│
├── part1-conversational-ai/
│   ├── src/
│   ├── tests/
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── .env.example
│   └── README.md
│
├── part2-machine-learning/
│   ├── data/
│   │   └── appointments.csv
│   ├── models/
│   │   └── no_show_model.joblib
│   ├── results/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.csv
│   │   └── metrics.json
│   ├── train.py
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
```

---

# Part 1 — Conversational / Agentic AI

## Overview

Part 1 implements a conversational medical clinic assistant that processes patient requests and determines the appropriate next action.

The assistant can:

* Identify the user's intent
* Extract appointment information
* Detect missing information
* Determine the appropriate action
* Request confirmation before appointment changes
* Handle ambiguous requests safely
* Escalate to a human when appropriate

## Supported Intents

The assistant supports:

* Booking an appointment
* Rescheduling an appointment
* Cancelling an appointment
* Checking opening hours
* Checking doctor availability
* Human handoff
* Unclear or unsupported requests

## Safety and Confirmation

A key design principle is that appointment-changing actions should not be executed without appropriate confirmation.

For example:

> Book an appointment with Dr. George on Monday at 10 AM.

The assistant identifies the booking intent and extracts the available appointment information, but requires confirmation before executing the appointment action.

For an ambiguous request such as:

> I might want to see Dr. George tomorrow at 4, but don't book anything yet.

The assistant does not create an appointment and instead handles the request safely without executing a booking action.

## Testing

The Part 1 test suite contains **10 tests**.

```text
10 passed
```

Run the tests with:

```powershell
cd part1-conversational-ai
.venv\Scripts\activate
pytest -q
```

To run the assistant:

```powershell
python -m src.agent
```

More implementation details are available in:

`part1-conversational-ai/README.md`

---

# Part 2 — Machine Learning

## Overview

Part 2 implements a machine learning pipeline for predicting whether a patient is likely to miss a scheduled appointment.

The workflow includes:

* Dataset validation
* Exploratory data inspection
* Missing-value checking
* Duplicate detection
* Feature selection
* Numerical and categorical preprocessing
* Train/test splitting
* Model training
* Model comparison
* Evaluation using multiple metrics
* Feature importance analysis
* Model persistence
* Example predictions

## Dataset

The assessment provided a historical appointment dataset containing **3,000 appointment records**.

The dataset contains the following 12 columns:

| Feature                   | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `appointment_id`          | Unique appointment identifier                  |
| `age`                     | Patient age                                    |
| `gender`                  | Patient gender                                 |
| `appointment_type`        | Type of appointment                            |
| `days_before_appointment` | Number of days between booking and appointment |
| `previous_appointments`   | Number of previous appointments                |
| `previous_no_shows`       | Number of previous no-shows                    |
| `weekday`                 | Appointment weekday                            |
| `appointment_time`        | Appointment time range                         |
| `reminder_sent`           | Whether a reminder was sent                    |
| `new_patient`             | Whether the patient is new                     |
| `no_show`                 | Target variable                                |

The `appointment_id` field is treated only as an identifier and is **not used as a predictive feature**.

### Dataset Validation

The official dataset contains:

* **3,000 records**
* **0 duplicate rows**
* **0 missing values**

### Target Distribution

| Outcome  | Count | Percentage |
| -------- | ----: | ---------: |
| Attended | 2,522 |     84.07% |
| No-show  |   478 |     15.93% |

The target is therefore imbalanced, making metrics such as **precision, recall, and F1-score for the no-show class** particularly important.

---

## Features

The predictive features are divided into:

### Numerical Features

* `age`
* `days_before_appointment`
* `previous_appointments`
* `previous_no_shows`
* `reminder_sent`
* `new_patient`

Numerical values are processed using median imputation and standardization.

### Categorical Features

* `gender`
* `appointment_type`
* `weekday`
* `appointment_time`

Categorical values are processed using most-frequent imputation and one-hot encoding.

Unknown categorical values are handled safely during prediction.

---

## Train/Test Split

The dataset is divided using an **80/20 stratified split**:

```text
Training samples: 2,400
Testing samples:    600
```

A fixed random state of `42` is used to make the experiment reproducible.

---

# Models

Two classification models were evaluated:

1. **Logistic Regression**
2. **Random Forest**

The models were compared using:

* Accuracy
* Precision for the no-show class
* Recall for the no-show class
* F1-score for the no-show class
* ROC-AUC

Because the goal is to identify potential no-shows, the **F1-score for the no-show class** was used as the primary model-selection metric.

---

# Results

## Logistic Regression

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 0.8517 |
| Precision (No-show) | 0.6667 |
| Recall (No-show)    | 0.1458 |
| F1-score (No-show)  | 0.2393 |
| ROC-AUC             | 0.6959 |

## Random Forest

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 0.8100 |
| Precision (No-show) | 0.3816 |
| Recall (No-show)    | 0.3021 |
| F1-score (No-show)  | 0.3372 |
| ROC-AUC             | 0.6840 |

---

# Selected Model

**Random Forest** was selected as the final model because it achieved the higher **F1-score for the no-show class**:

```text
Logistic Regression: 0.2393
Random Forest:       0.3372
```

Although Logistic Regression achieved higher overall accuracy and ROC-AUC, Random Forest identified a larger proportion of actual no-shows, resulting in a better F1-score for the target class.

The trained pipeline is saved as:

```text
models/no_show_model.joblib
```

---

# Feature Importance

The Random Forest model identified the following features as the most influential:

| Feature                 | Importance |
| ----------------------- | ---------: |
| Age                     |     0.1827 |
| Days before appointment |     0.1521 |
| Previous appointments   |     0.1246 |
| Previous no-shows       |     0.1190 |
| Reminder sent           |     0.0278 |

The complete feature importance results are available in:

```text
results/feature_importance.csv
```

---

# Example Predictions

The prediction script loads the trained model and generates no-show probabilities for example appointments.

Example output:

```text
age  appointment_type   days_before_appointment  previous_no_shows  reminder_sent  no_show_probability
35   Follow-up          7                        0                  1              0.057
52   New Consultation   30                       0                  0              0.403
61   Routine Check      3                        0                  1              0.153
```

The prediction script converts these probabilities into a simple interpretation such as:

```text
Likely to attend
```

or

```text
Likely to miss appointment
```

The default classification threshold is `0.5`.

---

# Results and Artifacts

The training pipeline generates:

```text
models/
└── no_show_model.joblib

results/
├── confusion_matrix.png
├── feature_importance.csv
└── metrics.json
```

These artifacts provide the trained model, evaluation results, feature importance, and confusion matrix for inspection.

---

# Running Part 2

From the project root:

```powershell
cd part2-machine-learning
.venv\Scripts\activate
```

Train the models:

```powershell
python train.py
```

Generate example predictions:

```powershell
python predict.py
```

The official dataset is already included in:

```text
data/appointments.csv
```

No dataset-generation step is required.

---

# Technologies

## Part 1

* Python
* Pytest
* Rule-based conversational logic
* Structured outputs
* Environment-based configuration

## Part 2

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

# Testing

Both parts were tested locally.

### Part 1

```text
10 passed
```

### Part 2

The complete ML workflow was successfully executed:

```powershell
python train.py
python predict.py
```

The training process successfully generated:

* Trained Random Forest model
* Model comparison metrics
* Feature importance
* Confusion matrix
* Example predictions

---

# Limitations and Responsible Use

This repository is an educational assessment project and should not be considered a production-ready clinical system.

The machine learning model is trained on the dataset provided for the assessment. Its results should not be interpreted as validated clinical performance or as a substitute for medical decision-making.

For production deployment, additional work would be required, including:

* Larger and more representative datasets
* External validation
* Monitoring for data and model drift
* Fairness and bias evaluation
* Appropriate privacy and security controls
* Threshold optimization based on operational requirements
* Human oversight
* Integration testing with clinical systems

The conversational assistant is also a prototype and would require additional validation, security, privacy controls, and integration testing before any real-world medical use.

---

# Reproducibility

The ML pipeline uses a fixed random state and a complete preprocessing/model pipeline to make training and prediction reproducible.

The main workflow is:

```text
Official Dataset
       ↓
Data Validation
       ↓
Feature Selection
       ↓
Preprocessing
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Selection
       ↓
Saved Model
       ↓
Example Predictions
```

---

# Author

**Rony Ghanem**

Master's in Management Information Systems

* GitHub: `github.com/ronyghanem`
* LinkedIn: `linkedin.com/in/rony-ghanem`
* Portfolio: `ronygh.netlify.app`
