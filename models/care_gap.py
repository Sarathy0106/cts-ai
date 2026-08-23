"""
Care Gap Analysis Model
Identifies clinical care gaps based on HEDIS-inspired quality measures.
Rule-based — no ML training required.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Any


# Each measure: (display_name, check_fn, severity)
# check_fn(patient) -> True means gap EXISTS (measure not met)
_MEASURES = []


def _age_between(patient, lo, hi):
    age = int(patient.get("age", 0))
    return lo <= age <= hi


def _missing(patient, key):
    val = patient.get(key)
    return val is None or val is False or val == "" or val == []


def _days_since(patient, key) -> int | None:
    """Return days since a date string (YYYY-MM-DD), or None if missing."""
    val = patient.get(key)
    if not val:
        return None
    try:
        d = date.fromisoformat(str(val))
        return (date.today() - d).days
    except ValueError:
        return None


# ── measure definitions ────────────────────────────────────────────────────────

MEASURE_CATALOG: list[dict] = [
    {
        "id": "BCS",
        "name": "Breast Cancer Screening",
        "description": "Women 50–74 should have mammogram within 2 years",
        "severity": "high",
        "check": lambda p: (
            p.get("sex", "").lower() == "female"
            and _age_between(p, 50, 74)
            and (
                _missing(p, "last_mammogram_date")
                or (_days_since(p, "last_mammogram_date") or 9999) > 730
            )
        ),
    },
    {
        "id": "CCS",
        "name": "Cervical Cancer Screening",
        "description": "Women 21–64 should have Pap smear within 3 years",
        "severity": "high",
        "check": lambda p: (
            p.get("sex", "").lower() == "female"
            and _age_between(p, 21, 64)
            and (
                _missing(p, "last_pap_date")
                or (_days_since(p, "last_pap_date") or 9999) > 1095
            )
        ),
    },
    {
        "id": "CDC_A1C",
        "name": "Diabetes HbA1c Control",
        "description": "Diabetic patients should have HbA1c < 9% (tested within 1 year)",
        "severity": "critical",
        "check": lambda p: (
            "diabetes" in [c.lower() for c in p.get("conditions", [])]
            and (
                _missing(p, "last_a1c_date")
                or (_days_since(p, "last_a1c_date") or 9999) > 365
                or float(p.get("last_a1c_value", 0) or 0) >= 9.0
            )
        ),
    },
    {
        "id": "CBP",
        "name": "Controlling High Blood Pressure",
        "description": "Hypertensive patients 18–85 should have BP < 140/90",
        "severity": "high",
        "check": lambda p: (
            "hypertension" in [c.lower() for c in p.get("conditions", [])]
            and _age_between(p, 18, 85)
            and (
                _missing(p, "last_bp_systolic")
                or int(p.get("last_bp_systolic", 0) or 0) >= 140
                or int(p.get("last_bp_diastolic", 0) or 0) >= 90
            )
        ),
    },
    {
        "id": "COL",
        "name": "Colorectal Cancer Screening",
        "description": "Adults 50–75 should have colorectal screening within 10 years",
        "severity": "high",
        "check": lambda p: (
            _age_between(p, 50, 75)
            and (
                _missing(p, "last_colonoscopy_date")
                or (_days_since(p, "last_colonoscopy_date") or 9999) > 3650
            )
        ),
    },
    {
        "id": "FLU",
        "name": "Influenza Immunization",
        "description": "Annual flu vaccine recommended",
        "severity": "moderate",
        "check": lambda p: (
            _missing(p, "last_flu_vaccine_date")
            or (_days_since(p, "last_flu_vaccine_date") or 9999) > 365
        ),
    },
    {
        "id": "MED_REC",
        "name": "Medication Reconciliation",
        "description": "Patients on 5+ medications should have med review within 1 year",
        "severity": "moderate",
        "check": lambda p: (
            int(p.get("medications_count", 0) or 0) >= 5
            and (
                _missing(p, "last_med_reconciliation_date")
                or (_days_since(p, "last_med_reconciliation_date") or 9999) > 365
            )
        ),
    },
    {
        "id": "AWV",
        "name": "Annual Wellness Visit",
        "description": "Medicare patients 65+ should have annual wellness visit",
        "severity": "moderate",
        "check": lambda p: (
            int(p.get("age", 0) or 0) >= 65
            and (
                _missing(p, "last_wellness_visit_date")
                or (_days_since(p, "last_wellness_visit_date") or 9999) > 365
            )
        ),
    },
    {
        "id": "DEP_SCR",
        "name": "Depression Screening",
        "description": "Adults 18+ should have depression screening within 1 year",
        "severity": "moderate",
        "check": lambda p: (
            _age_between(p, 18, 120)
            and (
                _missing(p, "last_depression_screen_date")
                or (_days_since(p, "last_depression_screen_date") or 9999) > 365
            )
        ),
    },
    {
        "id": "KED",
        "name": "Kidney Health Evaluation (Diabetes)",
        "description": "Diabetics should have annual urine albumin and eGFR test",
        "severity": "high",
        "check": lambda p: (
            "diabetes" in [c.lower() for c in p.get("conditions", [])]
            and (
                _missing(p, "last_kidney_eval_date")
                or (_days_since(p, "last_kidney_eval_date") or 9999) > 365
            )
        ),
    },
]


def analyze_care_gaps(patient: dict[str, Any]) -> dict[str, Any]:
    """
    Returns:
      {
        "gaps_found": [ { "id", "name", "description", "severity" }, ... ],
        "gaps_met":   [ { "id", "name" }, ... ],
        "total_measures_checked": int,
        "gap_count": int,
        "highest_severity": str,   # critical / high / moderate / low / none
        "model": "care_gap_analysis"
      }
    """
    gaps_found = []
    gaps_met   = []

    severity_rank = {"critical": 4, "high": 3, "moderate": 2, "low": 1, "none": 0}

    for m in MEASURE_CATALOG:
        try:
            has_gap = m["check"](patient)
        except Exception:
            has_gap = False

        if has_gap:
            gaps_found.append({
                "id":          m["id"],
                "name":        m["name"],
                "description": m["description"],
                "severity":    m["severity"],
            })
        else:
            gaps_met.append({"id": m["id"], "name": m["name"]})

    highest = "none"
    for g in gaps_found:
        if severity_rank.get(g["severity"], 0) > severity_rank.get(highest, 0):
            highest = g["severity"]

    return {
        "gaps_found":             gaps_found,
        "gaps_met":               gaps_met,
        "total_measures_checked": len(MEASURE_CATALOG),
        "gap_count":              len(gaps_found),
        "highest_severity":       highest,
        "model":                  "care_gap_analysis",
    }
