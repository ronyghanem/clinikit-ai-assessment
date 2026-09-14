from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "no_show_model.joblib"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run `python train.py` first."
        )

    model = joblib.load(MODEL_PATH)

    # Example appointments using the same
    # data format as the official CliniKit dataset.
    examples = pd.DataFrame(
        [
            {
                "age": 35,
                "gender": "Female",
                "appointment_type": "Follow-up",
                "days_before_appointment": 7,
                "previous_appointments": 3,
                "previous_no_shows": 0,
                "weekday": "Tuesday",
                "appointment_time": "10:00-12:00",
                "reminder_sent": 1,
                "new_patient": 0,
            },
            {
                "age": 52,
                "gender": "Male",
                "appointment_type": "New Consultation",
                "days_before_appointment": 30,
                "previous_appointments": 0,
                "previous_no_shows": 0,
                "weekday": "Saturday",
                "appointment_time": "16:00-18:00",
                "reminder_sent": 0,
                "new_patient": 1,
            },
            {
                "age": 61,
                "gender": "Female",
                "appointment_type": "Routine Check",
                "days_before_appointment": 3,
                "previous_appointments": 8,
                "previous_no_shows": 0,
                "weekday": "Wednesday",
                "appointment_time": "14:00-16:00",
                "reminder_sent": 1,
                "new_patient": 0,
            },
        ]
    )

    probabilities = model.predict_proba(examples)[:, 1]

    predictions = model.predict(examples)

    results = examples.copy()

    results["no_show_probability"] = probabilities.round(3)

    results["predicted_no_show"] = predictions

    results["prediction"] = results[
        "predicted_no_show"
    ].map(
        {
            0: "Likely to attend",
            1: "Likely to miss appointment",
        }
    )

    print("=" * 80)
    print("CliniKit - Example No-Show Predictions")
    print("=" * 80)

    print(
        results[
            [
                "age",
                "appointment_type",
                "days_before_appointment",
                "previous_no_shows",
                "reminder_sent",
                "no_show_probability",
                "prediction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()