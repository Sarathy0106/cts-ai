import json
from data_loader import load_patients_csv
from ml_client import predict_batch, predict_single
from agent import run_agent

def main():
    print("==================================================")
    print("AI INTERVENTION AGENT: Full Orchestration Pipeline")
    print("==================================================")
    
    # 1. Load patients CSV
    csv_path = "../data/patients.csv"
    try:
        df = load_patients_csv(csv_path)
        print(f"Loaded {len(df)} patients from CSV.\n")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Retrieve ML predictions via Batch API
    print("Fetching batch predictions from ML Model API...")
    try:
        batch_output = predict_batch(csv_path)
        print("Batch predictions successfully retrieved.\n")
    except Exception as e:
        print(f"Batch prediction failure: {e}")
        return

    # 3. Iterate through batch predictions and pass directly to Agent
    predictions = batch_output.get("predictions", [])
    if not predictions:
        print("No predictions found in the ML model output.")
        return

    for idx, ml_res in enumerate(predictions):
        print(f"--------------------------------------------------")
        print(f"Processing Patient {idx+1}/{len(predictions)}: ID {ml_res.get('patient_id')}")
        print(f"ML Prediction: {ml_res.get('prediction')} ({ml_res.get('risk_level')} risk)")
        print(f"--------------------------------------------------")
        
        # Run the AI Intervention Agent on the ML result
        try:
            intervention_plan = run_agent(ml_res)
            print("\nFINAL AGENT STRUCTURED OUTPUT:")
            print(json.dumps(intervention_plan, indent=2))
            print("\n")
        except Exception as e:
            print(f"Agent failed to run for patient: {e}\n")

if __name__ == "__main__":
    main()