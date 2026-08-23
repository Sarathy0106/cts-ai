"""
Star Impact Model tool
Medicare Advantage Quality Star-Impact Simulator
Endpoint: GET https://api.sidanex.com/cts-ml4/simulate/{measure_code}/scenarios

Schema confirmed from https://api.sidanex.com/cts-ml4/openapi.json
  GET /simulate/{measure_code}/scenarios — runs full 8-scenario comparison,
  no request body needed. Returns list[DirectionalStarImpactOutput].
  We use the measure_code mapped from the patient's highest-severity care gap.
  Falls back to "C14" (Diabetes HbA1c Control) as the default measure code
  when no gap mapping is available.
  No auth required (confirmed 200 without credentials).

DirectionalStarImpactOutput fields (all required in response):
  current_rate, projected_rate_median, projected_rate_p05, projected_rate_p95,
  expected_gap_closures, current_star, projected_star, next_star_cut_point,
  probability_of_next_star, expected_cost, cost_per_expected_closure,
  scenario, recommendation
"""
from __future__ import annotations
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL  = "https://api.sidanex.com/cts-ml4"
TIMEOUT   = 30.0
MODEL_KEY = "star_impact_model"

# Map local care gap IDs → CMS measure codes used by the Star Impact API
# Based on the HEDIS-to-CMS alignment in the spec description
GAP_TO_MEASURE: dict[str, str] = {
    "CDC_A1C": "C14",   # Diabetes HbA1c Control
    "CBP":     "C18",   # Controlling High Blood Pressure
    "COL":     "C16",   # Colorectal Cancer Screening
    "BCS":     "C17",   # Breast Cancer Screening
    "KED":     "C15",   # Kidney Health Evaluation for Patients with Diabetes
    "FLU":     "D10",   # Annual Flu Vaccine
    "AWV":     "C20",   # Annual Wellness Visit
    "DEP_SCR": "C19",   # Depression Screening
    "MED_REC": "D08",   # Medication Reconciliation
}
DEFAULT_MEASURE = "C14"


def _select_measure_code(patient_data: dict[str, Any]) -> str:
    """
    Picks the most clinically relevant CMS measure code from the patient's
    conditions, or falls back to the default.
    Uses conditions as a proxy when full gap results are not available here.
    """
    conditions = [c.lower() for c in patient_data.get("conditions", [])]
    if "diabetes" in conditions:
        return "C14"
    if "hypertension" in conditions:
        return "C18"
    return DEFAULT_MEASURE


async def call_star_impact_model(patient_data: dict[str, Any]) -> dict[str, Any]:
    """
    Medicare Advantage Quality Star-Impact Simulator
    GET https://api.sidanex.com/cts-ml4/simulate/{measure_code}/scenarios
    Returns 8-scenario Monte Carlo simulation results for the selected CMS measure.
    Falls back gracefully on any failure — never raises.
    """
    measure_code = _select_measure_code(patient_data)
    url = f"{BASE_URL}/simulate/{measure_code}/scenarios"

    logger.debug("%s request: GET %s (measure=%s)", MODEL_KEY, url, measure_code)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url)

        logger.debug("%s response: HTTP %s body=%.400s",
                     MODEL_KEY, resp.status_code, resp.text)

        if resp.status_code in (401, 403):
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
            "status":       "ok",
            "model":        MODEL_KEY,
            "measure_code": measure_code,
            "prediction":   resp.json(),   # list[DirectionalStarImpactOutput]
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
star_impact_model_tool = call_star_impact_model
