"""
Prioritization Model tool
Patient Prioritization & Outreach Optimization
Endpoint: GET https://api.sidanex.com/cts-ml3/member/{member_id}

Schema confirmed from https://api.sidanex.com/cts-ml3/openapi.json
  This API has no POST endpoints — all endpoints are GETs.
  Per-patient lookup: GET /member/{member_id}  (path param, no request body)
  Response schema is untyped ({}) in the spec — response shape discovered at runtime.
  No auth required (confirmed 200 without credentials).
"""
from __future__ import annotations
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL  = "https://api.sidanex.com/cts-ml3"
ENDPOINT  = "/member/{member_id}"   # GET — path param only
TIMEOUT   = 30.0
MODEL_KEY = "prioritization_model"


async def call_prioritization_model(patient_data: dict[str, Any]) -> dict[str, Any]:
    """
    Patient Prioritization & Outreach Optimization
    GET https://api.sidanex.com/cts-ml3/member/{member_id}
    Returns prioritization data for the patient by member_id.
    Falls back gracefully on any failure — never raises.
    """
    patient_id = str(patient_data.get("patient_id") or "unknown")
    url = f"{BASE_URL}/member/{patient_id}"

    logger.debug("%s request: GET %s", MODEL_KEY, url)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url)

        logger.debug("%s response: HTTP %s body=%.400s",
                     MODEL_KEY, resp.status_code, resp.text)

        if resp.status_code == 404:
            # Member not found in the portfolio — not an error, just no record
            return {
                "status":     "unavailable",
                "reason":     f"Member '{patient_id}' not found in prioritization portfolio (404)",
                "model":      MODEL_KEY,
                "prediction": None,
            }
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
prioritization_model_tool = call_prioritization_model
