"""
Quality Gap Analysis Agent — FastAPI application

Endpoints:
  POST /analyze/{patient_id}   — run full analysis for a patient
  GET  /health                 — service liveness + remote service status
"""
from __future__ import annotations
import logging
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.health_check import check_remote_services
from agent.orchestrator import run_analysis

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quality Gap Analysis Agent",
    description=(
        "Combines Risk Score, Care Gap, and Patient Segmentation models with "
        "Ollama-based cross-model reasoning to produce a structured Quality Gap Analysis."
    ),
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class PatientInput(BaseModel):
    # Demographics
    age:  int             = Field(..., ge=0, le=130, example=68)
    sex:  str             = Field("unknown", example="female")

    # Clinical
    conditions:             list[str] = Field(default_factory=list,
                                              example=["diabetes", "hypertension"])
    medications_count:      int       = Field(0, ge=0)
    ed_visits_90d:          int       = Field(0, ge=0)
    hospitalizations_180d:  int       = Field(0, ge=0)
    has_pcp:                bool      = Field(True)
    med_adherence_pct:      float     = Field(100.0, ge=0, le=100)
    smoker:                 bool      = Field(False)
    bmi:                    float     = Field(22.0, ge=10, le=80)

    # Screening dates (ISO format YYYY-MM-DD, omit if unknown)
    last_mammogram_date:          str | None = None
    last_pap_date:                str | None = None
    last_a1c_date:                str | None = None
    last_a1c_value:               float | None = None
    last_bp_systolic:             int | None = None
    last_bp_diastolic:            int | None = None
    last_colonoscopy_date:        str | None = None
    last_flu_vaccine_date:        str | None = None
    last_med_reconciliation_date: str | None = None
    last_wellness_visit_date:     str | None = None
    last_depression_screen_date:  str | None = None
    last_kidney_eval_date:        str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "age": 68,
                "sex": "female",
                "conditions": ["diabetes", "hypertension"],
                "medications_count": 6,
                "ed_visits_90d": 1,
                "hospitalizations_180d": 0,
                "has_pcp": True,
                "med_adherence_pct": 55.0,
                "smoker": False,
                "bmi": 31.2,
                "last_a1c_date": "2023-01-15",
                "last_a1c_value": 9.4,
                "last_bp_systolic": 148,
                "last_bp_diastolic": 92,
                "last_flu_vaccine_date": "2022-09-01",
                "last_mammogram_date": "2021-06-10",
            }
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/analyze/{patient_id}", response_model=dict)
async def analyze_patient(patient_id: str, body: PatientInput) -> dict[str, Any]:
    """
    Run full Quality Gap Analysis for a patient.

    Calls Risk Score, Care Gap, and Segmentation models locally.
    Calls Risk Prediction, Prioritization, and Star Impact models remotely.
    Feeds all outputs to Ollama for cross-model reasoning.
    """
    logger.info("POST /analyze/%s", patient_id)
    patient_data = body.model_dump()
    try:
        result = await run_analysis(patient_id, patient_data)
        return result
    except Exception as exc:
        logger.error("Unhandled error in /analyze/%s: %s", patient_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Returns liveness status and reachability of remote ML services.
    """
    services = await check_remote_services()
    return {
        "status": "ok",
        "remote_services": [
            {
                "name":      s.name,
                "url":       s.url,
                "reachable": s.reachable,
                "detail":    s.detail,
            }
            for s in services
        ],
    }
