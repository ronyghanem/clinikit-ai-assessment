from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "no_show_model.joblib"
)


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found.\n"
            "Run `python train.py` first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    examples = pd.DataFrame(
        [
            {
                "age": 32,
                "gender": "Female",
                "appointment_type": "Follow-up",
                "days_before_appointment": 7,
                "previous_appointments": 4,
                "previous_no_shows": 0,
                "weekday": "Tuesday",
                "appointment_time": "10:00",
                "reminder_sent": "Yes",
                "new_patient": "No",
            },
            {
                "age": 45,
                "gender": "Male",
                "appointment_type": "General",
                "days_before_appointment": 45,
                "previous_appointments": 2,
                "previous_no_shows": 2,
                "weekday": "Saturday",
                "appointment_time": "17:00",
                "reminder_sent": "No",
                "new_patient": "Yes",
            },
            {
                "age": 61,
                "gender": "Female",
                "appointment_type": "Specialist",
                "days_before_appointment": 3,
                "previous_appointments": 8,
                "previous_no_shows": 0,
                "weekday": "Wednesday",
                "appointment_time": "14:00",
                "reminder_sent": "Yes",
                "new_patient": "No",
            },
        ]
    )

    probabilities = (
        model.predict_proba(
            examples
        )[:, 1]
    )

    predictions = model.predict(
        examples
    )

    results = examples.copy()

    results[
        "no_show_probability"
    ] = probabilities.round(3)

    results[
        "predicted_no_show"
    ] = predictions

    results["prediction"] = (
        results[
            "predicted_no_show"
        ].map(
            {
                0: "Likely to attend",
                1: "Likely to miss appointment",
            }
        )
    )

    print("=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

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