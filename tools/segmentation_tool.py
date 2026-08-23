"""
Tool wrapper for Patient Segmentation model.
Requires risk score and care gap count, which are resolved from prior tool outputs.
"""
from __future__ import annotations
import logging
from typing import Any
from models.segmentation import segment_patient

logger = logging.getLogger(__name__)


async def segmentation_tool(
    patient_data: dict[str, Any],
    risk_score: int = 0,
    care_gap_count: int = 0,
) -> dict[str, Any]:
    """
    Runs the local Patient Segmentation model.
    Always returns a structured dict — never raises.
    """
    try:
        logger.debug("segmentation_tool: risk_score=%s care_gap_count=%s",
                     risk_score, care_gap_count)
        result = segment_patient(patient_data, risk_score, care_gap_count)
        result["status"] = "ok"
        logger.debug("segmentation_tool: segment=%s", result["segment_id"])
        return result
    except Exception as exc:
        logger.error("segmentation_tool error: %s", exc, exc_info=True)
        return {
            "status":        "error",
            "reason":        str(exc),
            "model":         "patient_segmentation",
            "segment_id":    None,
            "segment_label": None,
        }
