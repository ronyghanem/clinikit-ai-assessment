from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


warnings.filterwarnings("ignore")


RANDOM_STATE = 42

BASE_DIR = Path(__file__).parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "appointments.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "no_show_model.joblib"
)

METRICS_PATH = (
    BASE_DIR
    / "results"
    / "metrics.json"
)

FEATURES_PATH = (
    BASE_DIR
    / "results"
    / "feature_importance.csv"
)

CONFUSION_PATH = (
    BASE_DIR
    / "results"
    / "confusion_matrix.png"
)

TARGET = "no_show"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Run `python generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("1. DATA EXPLORATION")
    print("=" * 70)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print(
        f"\nDuplicate rows: {df.duplicated().sum()}"
    )

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())

    print("\nTarget distribution (%):")

    print(
        (
            df[TARGET]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    return df


def build_preprocessor(X):
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
        [
            (
                "numeric",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision_no_show": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall_no_show": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_no_show": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print("\n" + "=" * 70)
    print(f"{name.upper()} RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Attended (0)",
                "No-show (1)",
            ],
            zero_division=0,
        )
    )

    return metrics


def get_feature_importance(pipeline):
    preprocessor = (
        pipeline.named_steps["preprocessor"]
    )

    classifier = (
        pipeline.named_steps["classifier"]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    if hasattr(classifier, "coef_"):
        values = np.abs(
            classifier.coef_[0]
        )

        importance_type = (
            "absolute_coefficient"
        )

    else:
        values = (
            classifier.feature_importances_
        )

        importance_type = (
            "random_forest_importance"
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "importance_type": importance_type,
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    )


def main():

    df = load_data()

    # Remove exact duplicates.
    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    preprocessor = build_preprocessor(
        X_train
    )

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    logistic_pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    random_forest_pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\n" + "=" * 70)
    print("2. MODEL TRAINING")
    print("=" * 70)

    logistic_pipeline.fit(
        X_train,
        y_train,
    )

    random_forest_pipeline.fit(
        X_train,
        y_train,
    )

    logistic_metrics = evaluate_model(
        "Logistic Regression",
        logistic_pipeline,
        X_test,
        y_test,
    )

    rf_metrics = evaluate_model(
        "Random Forest",
        random_forest_pipeline,
        X_test,
        y_test,
    )

    # --------------------------------------------------
    # Select best model
    # --------------------------------------------------

    if (
        rf_metrics["f1_no_show"]
        >= logistic_metrics["f1_no_show"]
    ):
        best_model = random_forest_pipeline
        best_name = "Random Forest"
        best_metrics = rf_metrics

    else:
        best_model = logistic_pipeline
        best_name = "Logistic Regression"
        best_metrics = logistic_metrics

    print("\n" + "=" * 70)
    print(
        f"3. SELECTED MODEL: {best_name}"
    )
    print("=" * 70)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    all_metrics = {
        "selected_model": best_name,
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = get_feature_importance(
        best_model
    )

    importance.to_csv(
        FEATURES_PATH,
        index=False,
    )

    print(
        "\nTop 15 influential features:"
    )

    print(
        importance
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    predictions = best_model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Attended",
            "No-show",
        ],
    ).plot(ax=ax)

    ax.set_title(
        f"Confusion Matrix - {best_name}"
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_PATH,
        dpi=150,
    )

    plt.close(fig)

    print("\nSaved artifacts:")

    print(
        f"- Model: {MODEL_PATH}"
    )

    print(
        f"- Metrics: {METRICS_PATH}"
    )

    print(
        f"- Feature importance: "
        f"{FEATURES_PATH}"
    )

    print(
        f"- Confusion matrix: "
        f"{CONFUSION_PATH}"
    )

    print("\n" + "=" * 70)
    print("4. INTERPRETATION")
    print("=" * 70)

    print(
        "Recall for the no-show class is "
        "important because the clinic may "
        "want to identify as many potentially "
        "missed appointments as possible."
    )

    print(
        "ROC-AUC is useful because it measures "
        "how well the model ranks higher-risk "
        "patients against lower-risk patients."
    )


if __name__ == "__main__":
    main()