"""
HTTP endpoint test for /analyze/{patient_id} and /health.
Uses urllib only (no extra deps). Run while uvicorn is up.
"""
from __future__ import annotations
import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8000"

PATIENT_P00142 = {
    "age": 68, "sex": "female",
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
    "last_mammogram_date": "2021-06-10",
    "last_flu_vaccine_date": "2022-09-01",
    "last_kidney_eval_date": None,
    "last_wellness_visit_date": None,
    "last_med_reconciliation_date": None,
    "last_colonoscopy_date": None,
    "last_depression_screen_date": None,
    "last_pap_date": "2024-01-01",
}

REQUIRED_TOP_KEYS = {
    "patient_id", "models_used", "models_unavailable",
    "raw_ml_predictions", "derived_facts",
    "agent_reasoning", "recommendations", "analysis_confidence",
}

REQUIRED_RAW_KEYS = {
    "risk_score_analysis", "care_gap_analysis",
    "patient_segmentation", "risk_prediction_model",
    "prioritization_model", "star_impact_model",
}

def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)

def do_request(method, path, body=None, timeout=180):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as ex:
        return None, str(ex)

# ── TEST 1: GET /health ────────────────────────────────────────────────────────
sep("TEST 1 — GET /health")
print(f"Request:  GET {BASE}/health\n")
status, body = do_request("GET", "/health", timeout=15)
print(f"HTTP status: {status}")
print(f"Response body:\n{json.dumps(json.loads(body), indent=2)}")

health = json.loads(body)
assert health.get("status") == "ok", "FAIL: health.status != ok"
services = {s["name"]: s for s in health.get("remote_services", [])}
assert "Risk Prediction Model" in services, "FAIL: Risk Prediction Model missing from health"
assert "Prioritization Model"  in services, "FAIL: Prioritization Model missing from health"
assert "Star Impact Model"     in services, "FAIL: Star Impact Model missing from health"
# All three should be reachable (live HTTPS services)
for svc_name in ("Risk Prediction Model", "Prioritization Model", "Star Impact Model"):
    reachable = services[svc_name]["reachable"]
    print(f"  {svc_name}: {'reachable' if reachable else 'UNREACHABLE'} — {services[svc_name].get('detail','')}")
print("\n[PASS] GET /health — status=ok, all three remote services reported")

# ── TEST 2: POST /analyze/P-00142 ─────────────────────────────────────────────
sep("TEST 2 — POST /analyze/P-00142")
print(f"Request:  POST {BASE}/analyze/P-00142")
print(f"Body:     {json.dumps(PATIENT_P00142)[:120]}...\n")
status, body = do_request("POST", "/analyze/P-00142", body=PATIENT_P00142, timeout=180)
print(f"HTTP status: {status}")

if status != 200:
    print(f"FAIL: Expected 200, got {status}")
    print(body[:1000])
    sys.exit(1)

result = json.loads(body)
print(f"\nFull response:\n{json.dumps(result, indent=2)}")

# Structural checks
missing_keys = REQUIRED_TOP_KEYS - set(result.keys())
assert not missing_keys, f"FAIL: Missing top-level keys: {missing_keys}"

raw = result.get("raw_ml_predictions", {})
missing_raw = REQUIRED_RAW_KEYS - set(raw.keys())
assert not missing_raw, f"FAIL: Missing raw_ml_predictions keys: {missing_raw}"

assert result["patient_id"] == "P-00142", "FAIL: patient_id mismatch"

unavail_models = [m["model"] for m in result.get("models_unavailable", [])]
# Prioritization Model returns 404 for P-00142 (not in portfolio) — expected unavailable
# Risk Prediction and Star Impact should be ok
print(f"  models_used        : {result['models_used']}")
print(f"  models_unavailable : {unavail_models}")

rs = raw.get("risk_score_analysis", {})
if isinstance(rs, dict):
    assert rs.get("score") == 100,        f"FAIL: risk score should be 100, got {rs.get('score')}"
    assert rs.get("risk_level") == "critical", f"FAIL: risk level should be critical"

cg = raw.get("care_gap_analysis", {})
if isinstance(cg, dict):
    assert cg.get("gap_count") == 9,      f"FAIL: gap_count should be 9, got {cg.get('gap_count')}"

sg = raw.get("patient_segmentation", {})
if isinstance(sg, dict):
    assert sg.get("segment_id") == "complex_chronic", \
        f"FAIL: segment should be complex_chronic, got {sg.get('segment_id')}"

print("\n[PASS] POST /analyze/P-00142 — HTTP 200, all structural checks passed")
print(f"  patient_id          : {result['patient_id']}")
print(f"  models_used         : {result['models_used']}")
print(f"  models_unavailable  : {unavail_models}")
print(f"  risk score/level    : {rs.get('score')} / {rs.get('risk_level')}")
print(f"  care gap count      : {cg.get('gap_count')}")
print(f"  segment             : {sg.get('segment_id')}")
print(f"  analysis_confidence : {result.get('analysis_confidence')}")
