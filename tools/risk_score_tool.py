"""
Tool wrapper for Risk Score Analysis model.
"""
from __future__ import annotations
import logging
from typing import Any
from models.risk_score import compute_risk_score

logger = logging.getLogger(__name__)


async def risk_score_tool(patient_data: dict[str, Any]) -> dict[str, Any]:
    """
    Runs the local Risk Score Analysis model.
    Always returns a structured dict — never raises.
    """
    try:
        logger.debug("risk_score_tool: input keys=%s", list(patient_data.keys()))
        result = compute_risk_score(patient_data)
        result["status"] = "ok"
        logger.debug("risk_score_tool: score=%s level=%s", result["score"], result["risk_level"])
        return result
    except Exception as exc:
        logger.error("risk_score_tool error: %s", exc, exc_info=True)
        return {
            "status":  "error",
            "reason":  str(exc),
            "model":   "risk_score_analysis",
            "score":   None,
            "risk_level": None,
        }
