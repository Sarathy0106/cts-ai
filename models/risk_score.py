"""
Risk Score Analysis Model
Computes a composite risk score from patient clinical and demographic data.
Rule/heuristic-based — no training required.
"""
from __future__ import annotations
from typing import Any


# Weight table for risk factors
_RISK_WEIGHTS = {
    "age_gte_65":           15,
    "age_gte_75":           10,   # additive on top of age_gte_65
    "diabetes":             20,
    "hypertension":         15,
    "heart_disease":        25,
    "copd":                 20,
    "ckd":                  22,
    "cancer":               25,
    "obesity":              12,
    "smoking":              10,
    "depression":           10,
    "multiple_medications":  8,   # polypharmacy >= 5 meds
    "recent_ed_visit":      18,   # ED visit in last 90 days
    "recent_hospitalization":20,  # hospitalization in last 180 days
    "no_pcp":               12,   # no primary care provider
    "low_adherence":        15,   # medication adherence < 60%
}


def compute_risk_score(patient: dict[str, Any]) -> dict[str, Any]:
    """
    Input keys (all optional, treated as False/0 if absent):
      age                  : int
      conditions           : list[str]  e.g. ["diabetes","hypertension"]
      medications_count    : int
      ed_visits_90d        : int
      hospitalizations_180d: int
      has_pcp              : bool
      med_adherence_pct    : float  0-100
      smoker               : bool
      bmi                  : float

    Returns:
      {
        "score": int,           # 0–100 clamped
        "risk_level": str,      # low / moderate / high / critical
        "factors_present": list[str],
        "factors_checked": list[str],
        "model": "risk_score_analysis"
      }
    """
    age        = int(patient.get("age", 0))
    conditions = [c.lower().strip() for c in patient.get("conditions", [])]
    med_count  = int(patient.get("medications_count", 0))
    ed_visits  = int(patient.get("ed_visits_90d", 0))
    hosp       = int(patient.get("hospitalizations_180d", 0))
    has_pcp    = bool(patient.get("has_pcp", True))
    adherence  = float(patient.get("med_adherence_pct", 100.0))
    smoker     = bool(patient.get("smoker", False))
    bmi        = float(patient.get("bmi", 22.0))

    def has(cond: str) -> bool:
        return cond in conditions

    flags: dict[str, bool] = {
        "age_gte_65":            age >= 65,
        "age_gte_75":            age >= 75,
        "diabetes":              has("diabetes"),
        "hypertension":          has("hypertension"),
        "heart_disease":         has("heart_disease") or has("chf") or has("cad"),
        "copd":                  has("copd"),
        "ckd":                   has("ckd") or has("chronic kidney disease"),
        "cancer":                has("cancer"),
        "obesity":               bmi >= 30.0,
        "smoking":               smoker,
        "depression":            has("depression") or has("anxiety"),
        "multiple_medications":  med_count >= 5,
        "recent_ed_visit":       ed_visits >= 1,
        "recent_hospitalization":hosp >= 1,
        "no_pcp":                not has_pcp,
        "low_adherence":         adherence < 60.0,
    }

    raw_score      = sum(_RISK_WEIGHTS[k] for k, v in flags.items() if v)
    score          = min(100, raw_score)
    factors_present = [k for k, v in flags.items() if v]

    if score >= 70:
        risk_level = "critical"
    elif score >= 45:
        risk_level = "high"
    elif score >= 20:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "score":           score,
        "risk_level":      risk_level,
        "factors_present": factors_present,
        "factors_checked": list(flags.keys()),
        "model":           "risk_score_analysis",
    }
