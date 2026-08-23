"""
Tool wrapper for Care Gap Analysis model.
"""
from __future__ import annotations
import logging
from typing import Any
from models.care_gap import analyze_care_gaps

logger = logging.getLogger(__name__)


async def care_gap_tool(patient_data: dict[str, Any]) -> dict[str, Any]:
    """
    Runs the local Care Gap Analysis model.
    Always returns a structured dict — never raises.
    """
    try:
        logger.debug("care_gap_tool: input keys=%s", list(patient_data.keys()))
        result = analyze_care_gaps(patient_data)
        result["status"] = "ok"
        logger.debug("care_gap_tool: gap_count=%s highest=%s",
                     result["gap_count"], result["highest_severity"])
        return result
    except Exception as exc:
        logger.error("care_gap_tool error: %s", exc, exc_info=True)
        return {
            "status":     "error",
            "reason":     str(exc),
            "model":      "care_gap_analysis",
            "gaps_found": [],
            "gap_count":  None,
        }
