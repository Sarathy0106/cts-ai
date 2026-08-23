"""
Assembles the combined ML context object from all tool outputs.
This is the single source of truth passed to the reasoning prompt and
stored verbatim in raw_ml_predictions in the final output.
"""
from __future__ import annotations
from typing import Any

# Statuses that mean a tool did not return a usable prediction
_UNAVAILABLE_STATUSES = {"unavailable", "error", "not_implemented"}


def build_context(
    patient_id: str,
    patient_data: dict[str, Any],
    risk_result: dict[str, Any],
    care_gap_result: dict[str, Any],
    segment_result: dict[str, Any],
    risk_pred_result: dict[str, Any],
    priority_result: dict[str, Any],
    star_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Returns a structured context dict containing all tool outputs.
    This function is pure -- same inputs always produce the same output.
    """
    all_tools = {
        "risk_score_analysis":   risk_result,
        "care_gap_analysis":     care_gap_result,
        "patient_segmentation":  segment_result,
        "risk_prediction_model": risk_pred_result,
        "prioritization_model":  priority_result,
        "star_impact_model":     star_result,
    }

    models_available   = [k for k, v in all_tools.items() if v.get("status") == "ok"]
    models_unavailable = [
        {
            "model":  k,
            "status": v.get("status"),
            "reason": v.get("reason", "no reason returned"),
        }
        for k, v in all_tools.items()
        if v.get("status") in _UNAVAILABLE_STATUSES
    ]

    # ── Patient summary (stripped to clinical fields only) ────────────────────
    patient_summary = {
        "patient_id":             patient_id,
        "age":                    patient_data.get("age"),
        "sex":                    patient_data.get("sex"),
        "conditions":             patient_data.get("conditions", []),
        "medications_count":      patient_data.get("medications_count", 0),
        "ed_visits_90d":          patient_data.get("ed_visits_90d", 0),
        "hospitalizations_180d":  patient_data.get("hospitalizations_180d", 0),
        "has_pcp":                patient_data.get("has_pcp", True),
        "med_adherence_pct":      patient_data.get("med_adherence_pct", 100),
        "smoker":                 patient_data.get("smoker", False),
        "bmi":                    patient_data.get("bmi"),
    }

    # ── Derived facts (computed, not predicted) ───────────────────────────────
    derived_facts = []

    if risk_result.get("status") == "ok":
        derived_facts.append({
            "fact":   f"Risk score is {risk_result['score']}/100, classified as {risk_result['risk_level']}",
            "source": "risk_score_analysis",
        })
        if risk_result.get("factors_present"):
            derived_facts.append({
                "fact":   f"{len(risk_result['factors_present'])} risk factors active: {', '.join(risk_result['factors_present'])}",
                "source": "risk_score_analysis",
            })

    if care_gap_result.get("status") == "ok":
        derived_facts.append({
            "fact":   f"{care_gap_result['gap_count']} care gaps identified out of {care_gap_result['total_measures_checked']} measures checked",
            "source": "care_gap_analysis",
        })
        derived_facts.append({
            "fact":   f"Highest gap severity: {care_gap_result['highest_severity']}",
            "source": "care_gap_analysis",
        })
        critical_gaps = [g["id"] for g in care_gap_result.get("gaps_found", []) if g["severity"] == "critical"]
        if critical_gaps:
            derived_facts.append({
                "fact":   f"Critical gaps: {', '.join(critical_gaps)}",
                "source": "care_gap_analysis",
            })

    if segment_result.get("status") == "ok":
        derived_facts.append({
            "fact":   f"Patient segment: {segment_result['segment_id']} ({segment_result['segment_label']})",
            "source": "patient_segmentation",
        })

    if models_unavailable:
        derived_facts.append({
            "fact":   f"{len(models_unavailable)} model(s) unavailable: {', '.join(m['model'] for m in models_unavailable)}",
            "source": "health_check",
        })

    return {
        "patient_summary":    patient_summary,
        "models_available":   models_available,
        "models_unavailable": models_unavailable,
        "tool_outputs": {
            "risk_score_analysis":   risk_result,
            "care_gap_analysis":     care_gap_result,
            "patient_segmentation":  segment_result,
            "risk_prediction_model": risk_pred_result,
            "prioritization_model":  priority_result,
            "star_impact_model":     star_result,
        },
        "derived_facts": derived_facts,
    }
