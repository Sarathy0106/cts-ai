import requests
from config import ML_PREDICT_SINGLE_URL, ML_PREDICT_BATCH_URL

def predict_single(patient_data):
    """
    Sends single patient data as form data (application/x-www-form-urlencoded) to the ML API.
    """
    try:
        response = requests.post(
            ML_PREDICT_SINGLE_URL,
            data=patient_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Single Prediction API: {e}")
        raise

def predict_batch(csv_file_path):
    """
    Sends a POST request to predict_batch with a CSV file.
    """
    try:
        filename = csv_file_path.split("/")[-1].split("\\")[-1]
        with open(csv_file_path, "rb") as f:
            files = {"file": (filename, f, "text/csv")}
            response = requests.post(
                ML_PREDICT_BATCH_URL,
                files=files,
                timeout=60
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Batch Prediction API: {e}")
        raise
