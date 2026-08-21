import sys
import os
import json

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from agent import run_agent

sample_plan = {
    "plan_id": "PLAN-101",
    "current_stars": 3.5,
    "measures": [
        {
            "measure_code": "C01",
            "measure_name": "Breast Cancer Screening",
            "eligible": 1000,
            "completed": 700,
            "gaps": 300,
            "gaps_to_close": 150
        },
        {
            "measure_code": "C02",
            "measure_name": "Colorectal Cancer Screening",
            "eligible": 1200,
            "completed": 800,
            "gaps": 400,
            "gaps_to_close": 200
        }
    ]
}

if __name__ == "__main__":
    result = run_agent(sample_plan)
    print("\n--- FINAL RESULT ---")
    print(json.dumps(result, indent=2))
