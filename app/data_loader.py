import os
import pandas as pd

def load_patients_csv(csv_path="../data/patients.csv"):
    """
    Reads the patients CSV file, validates its existence and columns,
    and returns a pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Patients CSV file not found at: {os.path.abspath(csv_path)}")

    df = pd.read_csv(csv_path)

    # Map 'Id' column to 'patient_id' if present (new CSV uses 'Id')
    if "Id" in df.columns and "patient_id" not in df.columns:
        df = df.rename(columns={"Id": "patient_id"})

    # Validate essential columns
    required_cols = ["patient_id", "age"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {missing_cols}")

    # Safe conversions for numeric fields where necessary
    numeric_cols = [
        "age", "mpr", "pdc", "days_since_last_refill", "refill_frequency",
        "obesity_prevalence", "smoking_prevalence", "physical_inactivity",
        "diabetes_prevalence", "lack_of_insurance", "latest_bmi",
        "latest_systolic_bp", "latest_diastolic_bp", "latest_heart_rate",
        "latest_glucose", "latest_hba1c", "latest_cholesterol_total",
        "latest_cholesterol_hdl", "latest_body_weight", "latest_body_height",
        "encounter_duration_days", "medication_cost_per_rx"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
