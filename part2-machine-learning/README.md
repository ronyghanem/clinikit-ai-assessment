# Part 2 — Machine Learning: Appointment No-Show Prediction

## Overview

This project builds a machine learning system to predict whether a patient will miss a scheduled medical appointment.

The workflow includes:

1. Generating a synthetic appointment dataset.
2. Exploring the dataset.
3. Cleaning and preprocessing the data.
4. Handling missing values and duplicate records.
5. Training multiple machine learning models.
6. Evaluating model performance.
7. Selecting the best-performing model.
8. Saving the trained model and evaluation results.
9. Making predictions on new appointment examples.

> **Important:** No real patient dataset was provided with the assessment. Therefore, this project uses a synthetic dataset created specifically for this exercise. The results are for educational demonstration only and have no clinical validity.

---

## Dataset

The synthetic dataset contains **2,000 appointment records**.

The following features are included:

| Feature | Description |
|---|---|
| `age` | Patient age |
| `gender` | Patient gender |
| `appointment_type` | Type of appointment |
| `days_before_appointment` | Number of days between booking and appointment |
| `previous_appointments` | Number of previous appointments |
| `previous_no_shows` | Number of previous missed appointments |
| `weekday` | Appointment weekday |
| `appointment_time` | Appointment time |
| `reminder_sent` | Whether a reminder was sent |
| `new_patient` | Whether the patient is new |
| `no_show` | Target variable: 0 = attended, 1 = missed |

### Dataset Statistics

Target distribution:

- Attended: **1,553 (77.65%)**
- No-show: **447 (22.35%)**

The dataset also intentionally contains:

- 10 missing `age` values
- 10 missing `appointment_type` values
- 10 missing `appointment_time` values
- 1 duplicate row

These cases are included to demonstrate data-cleaning and preprocessing techniques.

---

## Project Structure

```text
part2-machine-learning/
│
├── data/
│   └── appointments.csv
│
├── models/
│   └── no_show_model.joblib
│
├── results/
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── confusion_matrix.png
│
├── generate_data.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
Installation

Create and activate a virtual environment.

Windows
python -m venv .venv
.venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
1. Generate the Dataset

Because no real dataset was provided, the project includes a script that generates a reproducible synthetic dataset.

Run:

python generate_data.py

The script creates:

data/appointments.csv

The dataset contains 2,000 synthetic appointment records.

2. Train the Models

Run:

python train.py

The training pipeline performs the following steps:

Data Exploration

The script checks:

Dataset shape
Data types
Missing values
Duplicate rows
Target distribution
Data Cleaning

Duplicate rows are removed before training.

Train/Test Split

The dataset is divided into:

80% training data
20% test data

A stratified split is used to preserve the target distribution.

Missing Value Handling

Numeric features use median imputation.

Categorical features use most-frequent-value imputation.

Feature Encoding

Categorical features are transformed using one-hot encoding.

Unknown categories are safely ignored during prediction.

Feature Scaling

Numeric features are standardized using StandardScaler.

Models

Two classification models are trained and compared:

1. Logistic Regression

Logistic Regression provides a simple and interpretable baseline model.

2. Random Forest

Random Forest is used as a non-linear tree-based model that can capture more complex relationships between features.

Evaluation Metrics

The models are evaluated using:

Accuracy
Precision for no-show
Recall for no-show
F1-score for no-show
ROC-AUC
Classification report
Confusion matrix

Because identifying patients who may miss appointments is important, the F1-score for the no-show class is used as the main model-selection metric.

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

Although Random Forest achieved higher overall accuracy and slightly higher ROC-AUC, Logistic Regression was preferred because the main objective is to identify the no-show class effectively.

Feature Influence

The trained model shows that several features contribute to the prediction, including:

Appointment time
Previous no-shows
Days before appointment
Previous appointments
New-patient status
Reminder status
Appointment weekday

The complete feature importance output is saved in:

results/feature_importance.csv
3. Make Predictions

After training the model, run:

python predict.py

The prediction script loads:

models/no_show_model.joblib

and predicts the probability that a patient will miss an appointment.

Example predictions from the trained model:

Example	No-show Probability	Prediction
Example 1	0.145	Likely to attend
Example 2	0.873	Likely to miss
Example 3	0.121	Likely to attend

The probability can be used by a clinic system to identify appointments that may require additional attention or reminders.

Generated Files

After training, the following files are created:

Trained Model
models/no_show_model.joblib

The selected Logistic Regression pipeline is saved using joblib.

Metrics
results/metrics.json

Contains the evaluation results for the trained models.

Feature Importance
results/feature_importance.csv

Contains the model's feature coefficients/influence.

Confusion Matrix
results/confusion_matrix.png

Provides a visual representation of the model's predictions.

Production Integration

A production clinic system could use the saved model as part of an appointment management workflow.

For example:

Patient books appointment
        ↓
Appointment information collected
        ↓
ML model predicts no-show probability
        ↓
Risk level calculated
        ↓
Clinic system decides whether
additional reminder actions are needed

The model should only support operational decisions and should not be used as a medical diagnosis tool.

Limitations

This project is an educational machine learning prototype.

Synthetic Data

The dataset is artificially generated because no real dataset was supplied with the assessment.

Therefore, the model's performance should not be interpreted as real-world clinical performance.

Limited Dataset Size

The dataset contains only 2,000 synthetic records and may not represent real patient behavior.

Moderate Performance

The ROC-AUC and F1-score are relatively modest.

This is expected given the synthetic dataset and simplified feature set.

Class Imbalance

The dataset contains more attended appointments than no-shows.

For this reason, accuracy alone is not sufficient to evaluate the model.

Real-World Deployment

Before production use, the model would require:

Real historical appointment data
Data validation
Larger datasets
Cross-validation
Hyperparameter tuning
Bias and fairness evaluation
Monitoring for model drift
Privacy and security controls
Clinical/operational validation
Reproducibility

The synthetic data generation uses a fixed random seed so that the experiment can be reproduced.

The complete workflow is:

python generate_data.py
python train.py
python predict.py
Technologies
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Joblib
Conclusion

This project demonstrates a complete machine learning workflow for appointment no-show prediction, including data generation, exploration, preprocessing, model training, evaluation, model selection, persistence, and prediction.

The solution prioritizes the no-show F1-score when selecting the final model and clearly documents the limitations of using synthetic data.

The resulting model is intended as an educational proof of concept rather than a production-ready clinical prediction system.