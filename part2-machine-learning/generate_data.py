from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_SAMPLES = 2000


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 81, N_SAMPLES)

    gender = rng.choice(
        ["Female", "Male"],
        N_SAMPLES,
        p=[0.55, 0.45],
    )

    appointment_type = rng.choice(
        ["General", "Follow-up", "Specialist", "Check-up"],
        N_SAMPLES,
        p=[0.40, 0.25, 0.20, 0.15],
    )

    days_before_appointment = rng.integers(
        1,
        61,
        N_SAMPLES,
    )

    previous_appointments = rng.poisson(
        2.5,
        N_SAMPLES,
    )

    previous_no_shows = np.array([
        rng.binomial(
            min(int(previous), 8),
            0.20,
        )
        if previous > 0
        else 0
        for previous in previous_appointments
    ])

    weekday = rng.choice(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ],
        N_SAMPLES,
        p=[0.18, 0.18, 0.18, 0.18, 0.20, 0.08],
    )

    appointment_time = rng.choice(
        [
            "09:00",
            "10:00",
            "11:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
        ],
        N_SAMPLES,
        p=[
            0.10,
            0.12,
            0.12,
            0.13,
            0.14,
            0.14,
            0.13,
            0.12,
        ],
    )

    reminder_sent = rng.choice(
        ["Yes", "No"],
        N_SAMPLES,
        p=[0.78, 0.22],
    )

    new_patient = rng.choice(
        ["Yes", "No"],
        N_SAMPLES,
        p=[0.25, 0.75],
    )

    # Synthetic relationship used only to create
    # demonstration labels.
    #
    # This is NOT real clinical data.
    score = (
        -2.05
        + 0.018 * days_before_appointment
        + 0.55 * previous_no_shows
        - 0.30 * (reminder_sent == "Yes")
        + 0.18 * (new_patient == "Yes")
        + 0.18 * (appointment_time >= "16:00")
        + 0.12 * (weekday == "Saturday")
        + 0.10 * (appointment_type == "General")
        - 0.04 * np.minimum(previous_appointments, 8)
        + rng.normal(0, 0.35, N_SAMPLES)
    )

    probability = 1 / (1 + np.exp(-score))

    no_show = rng.binomial(
        1,
        probability,
    )

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "appointment_type": appointment_type,
            "days_before_appointment": days_before_appointment,
            "previous_appointments": previous_appointments,
            "previous_no_shows": previous_no_shows,
            "weekday": weekday,
            "appointment_time": appointment_time,
            "reminder_sent": reminder_sent,
            "new_patient": new_patient,
            "no_show": no_show,
        }
    )

    # Add a small number of missing values to demonstrate
    # how the preprocessing pipeline handles them.
    for column in [
        "age",
        "appointment_type",
        "appointment_time",
    ]:
        indices = rng.choice(
            df.index,
            size=10,
            replace=False,
        )

        df.loc[indices, column] = np.nan

    output_path = (
        Path(__file__).parent
        / "data"
        / "appointments.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Generated {len(df)} synthetic appointments."
    )

    print(f"Saved to: {output_path}")

    print("\nTarget distribution:")
    print(df["no_show"].value_counts())

    print("\nTarget percentages:")
    print(
        (
            df["no_show"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )


if __name__ == "__main__":
    main()