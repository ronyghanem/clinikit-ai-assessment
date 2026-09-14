# Part 2 — Machine Learning: Appointment No-Show Prediction

## Overview

This project builds a machine learning system to predict whether a patient is likely to miss a scheduled medical appointment.

The solution uses the **official CliniKit dataset provided for this assessment**, containing 3,000 historical appointment records.

The workflow includes:

1. Loading and validating the provided dataset.
2. Exploring missing values, duplicates, and target distribution.
3. Removing the identifier column from model features.
4. Preprocessing numerical and categorical features.
5. Splitting the data using a stratified train/test split.
6. Training and comparing Logistic Regression and Random Forest models.
7. Evaluating model performance using classification metrics.
8. Selecting the best model based on no-show F1-score.
9. Saving the trained model and evaluation results.
10. Making predictions on new appointment examples.

> **Important:** The dataset was provided by CliniKit specifically for this assessment. The resulting model is an assessment prototype and should not be interpreted as clinically validated or suitable for medical decision-making.

---

## Dataset

The provided dataset contains **3,000 historical appointment records**, with appointment IDs ranging from **1 to 3000**.

The dataset contains the following fields:

| Feature                   | Description                                    |
| ------------------------- | ---------------------------------------------- |
| `appointment_id`          | Unique appointment identifier                  |
| `age`                     | Patient age                                    |
| `gender`                  | Patient gender                                 |
| `appointment_type`        | Type of appointment                            |
| `days_before_appointment` | Number of days between booking and appointment |
| `previous_appointments`   | Number of previous appointments                |
| `previous_no_shows`       | Number of previous missed appointments         |
| `weekday`                 | Appointment weekday                            |
| `appointment_time`        | Appointment time range                         |
| `reminder_sent`           | Whether a reminder was sent                    |
| `new_patient`             | Whether the patient is a new patient           |
| `no_show`                 | Target variable: 0 = attended, 1 = missed      |

### Dataset Validation

The dataset contains:

* **3,000 records**
* **12 columns**
* **0 duplicate rows**
* **0 missing values**

The `appointment_id` column is used only as an identifier and is excluded from model training.

### Target Distribution

| Outcome   |   Records | Percentage |
| --------- | --------: | ---------: |
| Attended  |     2,522 |     84.07% |
| No-show   |       478 |     15.93% |
| **Total** | **3,000** |   **100%** |

Because the no-show class represents only 15.93% of the dataset, accuracy alone is not sufficient for evaluating the model. Particular attention is therefore given to precision, recall, and F1-score for the no-show class.

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
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 1. Train the Models

The provided CliniKit dataset should be placed at:

```text
data/appointments.csv
```

Run:

```bash
python train.py
```

The training pipeline performs the following steps.

### Data Validation

The script checks:

* Dataset shape
* Required columns
* Missing values
* Duplicate rows
* Target distribution

### Feature Selection

`appointment_id` is excluded because it is an identifier and does not provide meaningful predictive information.

The model uses 10 predictive features:

**Numerical features:**

* `age`
* `days_before_appointment`
* `previous_appointments`
* `previous_no_shows`
* `reminder_sent`
* `new_patient`

**Categorical features:**

* `gender`
* `appointment_type`
* `weekday`
* `appointment_time`

### Train/Test Split

The dataset is divided into:

* **80% training data**
* **20% test data**

A stratified split with `random_state=42` is used to preserve the target distribution.

This produces:

* **2,400 training records**
* **600 testing records**

### Preprocessing

Numerical features are processed using:

* Median imputation
* StandardScaler

Categorical features are processed using:

* Most-frequent-value imputation
* One-hot encoding
* `handle_unknown="ignore"`

The preprocessing and model are stored together in a single scikit-learn pipeline to ensure consistent transformations during prediction.

---

# 2. Models

Two classification models are trained and compared.

## Logistic Regression

Logistic Regression provides a simple and interpretable baseline for binary classification.

## Random Forest

Random Forest is a non-linear ensemble model capable of capturing more complex relationships between appointment features.

Class balancing is enabled for the Random Forest to give additional weight to the minority no-show class.

---

# 3. Evaluation

The models are evaluated using:

* Accuracy
* Precision for no-show
* Recall for no-show
* F1-score for no-show
* ROC-AUC
* Classification report
* Confusion matrix

The **F1-score for the no-show class** is used as the primary model-selection metric because the main objective is to identify appointments that may be missed rather than simply maximize overall accuracy.

---

## Results

### Logistic Regression

| Metric              |  Score |
| ------------------- | -----: |
| Accuracy            | 0.8517 |
| Precision (No-show) | 0.6667 |
| Recall (No-show)    | 0.1458 |
| F1-score (No-show)  | 0.2393 |
| ROC-AUC             | 0.6959 |

### Random Forest

| Metric              |      Score |
| ------------------- | ---------: |
| Accuracy            |     0.8100 |
| Precision (No-show) |     0.3816 |
| Recall (No-show)    |     0.3021 |
| F1-score (No-show)  | **0.3372** |
| ROC-AUC             |     0.6840 |

---

## Selected Model

**Random Forest** was selected because it achieved the higher F1-score for the no-show class:

```text
Random Forest:       0.3372
Logistic Regression: 0.2393
```

Although Logistic Regression achieved higher overall accuracy and ROC-AUC, it detected substantially fewer actual no-shows, with a recall of only 14.58%.

Random Forest achieved:

* 30.21% no-show recall
* 38.16% no-show precision
* 33.72% no-show F1-score

This makes Random Forest the preferred model for the assessment objective of identifying potential no-shows.

---

# 4. Feature Importance

Feature influence is extracted from the selected Random Forest model after preprocessing.

The complete feature-importance output is saved in:

```text
results/feature_importance.csv
```

The output includes the transformed one-hot encoded categorical features as well as the numerical features.

This provides an indication of which input variables contributed most strongly to the model's predictions.

Feature importance should be interpreted as model behavior rather than causal evidence about why a patient may miss an appointment.

---

# 5. Make Predictions

After training the model, run:

```bash
python predict.py
```

The prediction script loads:

```text
models/no_show_model.joblib
```

and generates no-show probabilities for example appointments.

Example output from the trained model:

| Example   | No-show Probability | Prediction       |
| --------- | ------------------: | ---------------- |
| Example 1 |               0.057 | Likely to attend |
| Example 2 |               0.403 | Likely to attend |
| Example 3 |               0.153 | Likely to attend |

The probability represents the model's estimated likelihood of a no-show.

The default classification decision uses the model's standard probability threshold.

---

# 6. Generated Files

After running the training script, the following artifacts are produced.

### Trained Model

```text
models/no_show_model.joblib
```

Contains the complete preprocessing and Random Forest prediction pipeline.

### Metrics

```text
results/metrics.json
```

Contains the evaluation results for both trained models and the selected model.

### Feature Importance

```text
results/feature_importance.csv
```

Contains the feature importance values generated from the selected Random Forest model.

### Confusion Matrix

```text
results/confusion_matrix.png
```

Provides a visual representation of the model's predictions on the test set.

---

# 7. Example Workflow

The complete workflow is:

```bash
pip install -r requirements.txt
python train.py
python predict.py
```

No synthetic data generation step is required because the official CliniKit dataset is used directly.

---

# 8. Potential Production Integration

A clinic appointment-management system could use a similar model as an operational support tool.

For example:

```text
Patient books appointment
        ↓
Appointment information collected
        ↓
ML model predicts no-show probability
        ↓
Risk level calculated
        ↓
Clinic system determines whether
additional reminder actions are appropriate
```

The model should be treated as a decision-support component rather than a medical diagnostic system.

---

# 9. Limitations

### Assessment Dataset

The model was trained and evaluated on the dataset provided for this assessment. Its performance may not generalize to other clinics, patient populations, or appointment systems.

### Class Imbalance

Only 15.93% of appointments in the dataset are no-shows.

Therefore, accuracy alone can be misleading. The evaluation focuses on the no-show precision, recall, and F1-score.

### Moderate Predictive Performance

The selected Random Forest achieved:

* F1-score: **0.3372**
* Recall: **0.3021**
* ROC-AUC: **0.6840**

These results indicate that the model provides a useful baseline but is not sufficiently accurate for unsupervised real-world deployment.

### Limited Feature Set

The dataset contains a relatively small number of appointment-related variables. Additional operational features could potentially improve performance.

### Real-World Deployment

Before production use, the model would require:

* Larger and representative historical datasets
* Cross-validation
* Hyperparameter tuning
* Threshold optimization
* Bias and fairness evaluation
* Privacy and security controls
* Model monitoring
* Drift detection
* Operational validation
* Appropriate human oversight

---

# 10. Responsible Use

This model predicts appointment attendance behavior and should not be used to make medical diagnoses or clinical decisions.

A production implementation should use predictions only as an operational support signal, such as prioritizing reminders, while keeping appropriate human oversight.

Model predictions should also be handled according to applicable privacy and data-protection requirements.

---

# 11. Reproducibility

The training process uses:

```text
random_state = 42
```

for reproducible train/test splitting and model training.

The official dataset is stored at:

```text
data/appointments.csv
```

Running:

```bash
python train.py
```

recreates the trained model and evaluation artifacts.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow for appointment no-show prediction using the official CliniKit assessment dataset.

The solution includes:

* Dataset validation
* Data preprocessing
* Stratified train/test splitting
* Logistic Regression baseline
* Random Forest model
* Model comparison
* No-show-focused evaluation
* Model selection
* Feature importance analysis
* Model persistence
* Example predictions

The Random Forest model was selected based on its higher no-show F1-score of **0.3372**, compared with **0.2393** for Logistic Regression.

The resulting model is an **assessment prototype**, not a clinically validated prediction system.
