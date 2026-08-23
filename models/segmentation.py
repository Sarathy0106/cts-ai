"""
Patient Segmentation Model
Assigns patients to a segment based on risk, complexity, and care needs.
Rule-based — no ML training required.
"""
from __future__ import annotations
from typing import Any


# Segment definitions ordered by priority (most complex first)
_SEGMENTS = [
    {
        "id": "complex_chronic",
        "label": "Complex Chronic Care",
        "description": "2+ chronic conditions with poor control or any recent acute event",
        "priority": 1,
        "criteria": lambda f: (
            f["chronic_count"] >= 2
            and (f["recent_acute"] or f["poor_control"])
        ),
    },
    {
        "id": "high_utilizer",
        "label": "High Utilizer",
        "description": "Multiple ED visits or any hospitalization indicating care management gaps",
        "priority": 2,
        "criteria": lambda f: (
            f["ed_visits"] >= 2 or f["hospitalizations"] >= 1
        ),
    },
    {
        "id": "rising_risk",
        "label": "Rising Risk",
        "description": "1+ chronic condition with elevated risk score and no recent acute events",
        "priority": 3,
        "criteria": lambda f: (
            f["chronic_count"] >= 1
            and f["risk_score"] >= 30
            and not f["recent_acute"]
        ),
    },
    {
        "id": "preventive_gap",
        "label": "Preventive Care Gap",
        "description": "Relatively healthy but missing key preventive screenings",
        "priority": 4,
        "criteria": lambda f: (
            f["chronic_count"] == 0
            and f["care_gap_count"] >= 2
        ),
    },
    {
        "id": "healthy_engaged",
        "label": "Healthy & Engaged",
        "description": "Low risk, up-to-date on preventive care",
        "priority": 5,
        "criteria": lambda f: (
            f["risk_score"] < 20
            and f["care_gap_count"] <= 1
        ),
    },
    {
        "id": "general_population",
        "label": "General Population",
        "description": "Does not meet criteria for a higher-priority segment",
        "priority": 6,
        "criteria": lambda f: True,   # catch-all
    },
]


def _extract_features(patient: dict[str, Any], risk_score: int, care_gap_count: int) -> dict:
    conditions   = [c.lower().strip() for c in patient.get("conditions", [])]
    chronic_list = [
        "diabetes", "hypertension", "heart_disease", "chf", "cad",
        "copd", "ckd", "chronic kidney disease", "cancer",
        "depression", "anxiety", "obesity",
    ]
    chronic_count    = sum(1 for c in chronic_list if c in conditions)
    ed_visits        = int(patient.get("ed_visits_90d", 0) or 0)
    hospitalizations = int(patient.get("hospitalizations_180d", 0) or 0)
    recent_acute     = ed_visits >= 1 or hospitalizations >= 1
    # "poor control" = risk score high despite treatment
    poor_control     = risk_score >= 45

    return {
        "chronic_count":    chronic_count,
        "ed_visits":        ed_visits,
        "hospitalizations": hospitalizations,
        "recent_acute":     recent_acute,
        "poor_control":     poor_control,
        "risk_score":       risk_score,
        "care_gap_count":   care_gap_count,
    }


def segment_patient(
    patient: dict[str, Any],
    risk_score: int = 0,
    care_gap_count: int = 0,
) -> dict[str, Any]:
    """
    Args:
      patient        : raw patient dict
      risk_score     : output from compute_risk_score (score field)
      care_gap_count : output from analyze_care_gaps (gap_count field)

    Returns:
      {
        "segment_id":    str,
        "segment_label": str,
        "description":   str,
        "priority":      int,
        "features_used": dict,
        "model":         "patient_segmentation"
      }
    """
    features = _extract_features(patient, risk_score, care_gap_count)

    for seg in _SEGMENTS:
        try:
            if seg["criteria"](features):
                return {
                    "segment_id":    seg["id"],
                    "segment_label": seg["label"],
                    "description":   seg["description"],
                    "priority":      seg["priority"],
                    "features_used": features,
                    "model":         "patient_segmentation",
                }
        except Exception:
            continue

    # Should never reach here due to catch-all, but just in case
    return {
        "segment_id":    "unknown",
        "segment_label": "Unknown",
        "description":   "Segmentation failed",
        "priority":      99,
        "features_used": features,
        "model":         "patient_segmentation",
    }
