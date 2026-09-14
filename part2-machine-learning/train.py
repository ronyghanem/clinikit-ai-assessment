from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "appointments.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CliniKit - No-Show Prediction")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

# Remove duplicate rows if any exist.
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    print(f"Removing {duplicate_count} duplicate rows.")
    df = df.drop_duplicates()
else:
    print("Duplicate rows: 0")


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = [
    "appointment_id",
    "age",
    "gender",
    "appointment_type",
    "days_before_appointment",
    "previous_appointments",
    "previous_no_shows",
    "weekday",
    "appointment_time",
    "reminder_sent",
    "new_patient",
    "no_show",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["no_show"].value_counts())

print("\nTarget distribution (%):")
print(
    (df["no_show"].value_counts(normalize=True) * 100)
    .round(2)
)


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

# appointment_id is an identifier, not a predictive feature.
X = df.drop(columns=["no_show", "appointment_id"])
y = df["no_show"]


numeric_features = [
    "age",
    "days_before_appointment",
    "previous_appointments",
    "previous_no_shows",
    "reminder_sent",
    "new_patient",
]

categorical_features = [
    "gender",
    "appointment_type",
    "weekday",
    "appointment_time",
]


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
}


# ---------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------

results = {}
trained_models = {}

best_model_name = None
best_model = None
best_f1 = -1


for model_name, classifier in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {model_name}")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    results[model_name] = {
        "accuracy": round(float(accuracy), 4),
        "precision_no_show": round(float(precision), 4),
        "recall_no_show": round(float(recall), 4),
        "f1_no_show": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }

    trained_models[model_name] = pipeline

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Attended",
                "No-show",
            ],
            zero_division=0,
        )
    )

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_model = pipeline


# ---------------------------------------------------------
# Save best model
# ---------------------------------------------------------

model_path = MODEL_DIR / "no_show_model.joblib"

joblib.dump(best_model, model_path)

print("\n" + "=" * 60)
print(f"Best model: {best_model_name}")
print(f"Model saved to: {model_path}")
print("=" * 60)


# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------

metrics = {
    "dataset": {
        "records": int(len(df)),
        "features_used": int(X.shape[1]),
        "target": "no_show",
        "test_size": 0.20,
        "random_state": 42,
    },
    "models": results,
    "selected_model": best_model_name,
    "selection_metric": "f1_no_show",
}

metrics_path = RESULTS_DIR / "metrics.json"

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Confusion matrix for best model
# ---------------------------------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.imshow(cm)

ax.set_title(
    f"Confusion Matrix - {best_model_name}"
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Attended", "No-show"]
)

ax.set_yticklabels(
    ["Attended", "No-show"]
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.tight_layout()

confusion_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

preprocessor_fitted = best_model.named_steps["preprocessor"]
classifier = best_model.named_steps["classifier"]

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

if hasattr(classifier, "feature_importances_"):
    importance_values = classifier.feature_importances_

elif hasattr(classifier, "coef_"):
    importance_values = abs(
        classifier.coef_[0]
    )

else:
    importance_values = None


if importance_values is not None:

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance_path = (
        RESULTS_DIR / "feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False,
    )

    print(
        f"\nFeature importance saved to: "
        f"{feature_importance_path}"
    )


print(
    f"Metrics saved to: {metrics_path}"
)

print(
    f"Confusion matrix saved to: "
    f"{confusion_path}"
)

print("\nTraining completed successfully.")