"""
Risk Prediction Model tool
CMS Star Ratings and Clinical Health Risk Prediction
Endpoint: POST https://api.sidanex.com/cts-ml1/predict

Schema confirmed from https://api.sidanex.com/cts-ml1/openapi.json
  Request:  PatientProfile  (only `age` is required; all others optional)
  Response: PatientScoreResponse
              { member_id: str,
                predictions: [ ClinicalMeasureOutput, ... ] }
  No auth required (confirmed 200 without credentials).
"""
from __future__ import annotations
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.sidanex.com/cts-ml1"
ENDPOINT = "/predict"           # scores all 13 CMS measures in one call
TIMEOUT  = 30.0
MODEL_KEY = "risk_prediction_model"


def _build_payload(patient_data: dict[str, Any], patient_id: str) -> dict:
    """
    Maps local patient_data fields to the confirmed PatientProfile schema.
    All fields except `age` are optional — missing values fall back to API defaults.
    """
    conditions = [c.lower() for c in patient_data.get("conditions", [])]
    sex        = patient_data.get("sex", "").lower()

    return {
        "patient_id":               patient_id,
        "age":                      float(patient_data.get("age", 0)),
        "gender_M":                 1 if sex == "male" else 0,
        "encounter_count_12m":      float(patient_data.get("ed_visits_90d", 0)),
        "days_since_last_encounter": 0.0,
        "medication_unique_count":  float(patient_data.get("medications_count", 0)),
        "has_diabetes":             "diabetes" in conditions,
        "has_hypertension":         "hypertension" in conditions,
        "has_cvd":                  any(c in conditions for c in ("heart_disease", "cad", "cvd")),
        "has_heart_failure":        any(c in conditions for c in ("chf", "heart_failure")),
        "has_stroke":               "stroke" in conditions,
        "has_osteoporosis":         "osteoporosis" in conditions,
        "has_depression":           any(c in conditions for c in ("depression", "mdd")),
    }


async def call_risk_prediction_model(patient_data: dict[str, Any]) -> dict[str, Any]:
    """
    CMS Star Ratings and Clinical Health Risk Prediction
    POST https://api.sidanex.com/cts-ml1/predict
    Scores patient against all 13 CMS clinical quality measures.
    Returns structured result or graceful unavailable on any failure.
    """
    patient_id = patient_data.get("patient_id") or patient_data.get("age", "unknown")
    payload = _build_payload(patient_data, str(patient_id))
    url = f"{BASE_URL}{ENDPOINT}"

    logger.debug("%s request: url=%s payload=%s", MODEL_KEY, url, payload)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, json=payload)

        logger.debug("%s response: HTTP %s body=%.400s",
                     MODEL_KEY, resp.status_code, resp.text)

        if resp.status_code == 401 or resp.status_code == 403:
            return {
                "status":     "unavailable",
                "reason":     f"HTTP {resp.status_code} — authentication/authorization required",
                "model":      MODEL_KEY,
                "prediction": None,
            }
        if resp.status_code != 200:
            return {
                "status":     "unavailable",
                "reason":     f"HTTP {resp.status_code}: {resp.text[:200]}",
                "model":      MODEL_KEY,
                "prediction": None,
            }

        return {
            "status":     "ok",
            "model":      MODEL_KEY,
            "prediction": resp.json(),
        }

    except httpx.TimeoutException:
        logger.warning("%s timed out after %.0fs", MODEL_KEY, TIMEOUT)
        return {
            "status":     "unavailable",
            "reason":     f"Request timed out after {TIMEOUT}s",
            "model":      MODEL_KEY,
            "prediction": None,
        }
    except Exception as exc:
        logger.error("%s unexpected error: %s", MODEL_KEY, exc, exc_info=True)
        return {
            "status":     "unavailable",
            "reason":     f"{type(exc).__name__}: {exc}",
            "model":      MODEL_KEY,
            "prediction": None,
        }


# Registry alias
risk_prediction_model_tool = call_risk_prediction_model
