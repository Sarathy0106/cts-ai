"""
Smoke-test: verifies all imports, local models, and tools work correctly
without needing Ollama or remote services.
"""
import asyncio, json, sys

PATIENT = {
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
}

async def main():
    errors = []

    # 1. Risk Score
    try:
        from tools.risk_score_tool import risk_score_tool
        r = await risk_score_tool(PATIENT)
        assert r["status"] == "ok" and r["score"] is not None
        print(f"[OK] Risk Score: score={r['score']} level={r['risk_level']}")
        print(f"     Factors: {r['factors_present']}")
    except Exception as e:
        errors.append(f"risk_score_tool: {e}")
        print(f"[FAIL] risk_score_tool: {e}")

    # 2. Care Gap
    try:
        from tools.care_gap_tool import care_gap_tool
        c = await care_gap_tool(PATIENT)
        assert c["status"] == "ok"
        print(f"[OK] Care Gap: {c['gap_count']} gaps, highest={c['highest_severity']}")
        for g in c["gaps_found"]:
            print(f"     • [{g['severity'].upper()}] {g['id']}: {g['name']}")
    except Exception as e:
        errors.append(f"care_gap_tool: {e}")
        print(f"[FAIL] care_gap_tool: {e}")

    # 3. Segmentation — uses actual risk score and care gap count from above,
    #    not hardcoded overrides, so it tests the real pipeline inputs.
    try:
        from tools.segmentation_tool import segmentation_tool
        actual_risk_score  = r["score"] if r.get("status") == "ok" else 0
        actual_gap_count   = c["gap_count"] if c.get("status") == "ok" else 0
        s = await segmentation_tool(PATIENT, risk_score=actual_risk_score, care_gap_count=actual_gap_count)
        assert s["status"] == "ok"

        # Determinism check — call 5 more times with identical inputs, result must never change
        for _ in range(5):
            s2 = await segmentation_tool(PATIENT, risk_score=actual_risk_score, care_gap_count=actual_gap_count)
            assert s2["segment_id"] == s["segment_id"], (
                f"Non-deterministic segmentation: got {s2['segment_id']} then {s['segment_id']} "
                f"for identical inputs (risk={actual_risk_score}, gaps={actual_gap_count})"
            )

        print(f"[OK] Segmentation: {s['segment_id']} — {s['segment_label']}")
        print(f"     (determinism verified: 6 identical calls returned same segment)")
    except Exception as e:
        errors.append(f"segmentation_tool: {e}")
        print(f"[FAIL] segmentation_tool: {e}")

    # 4. Risk Prediction Model (remote — CMS Star Ratings)
    try:
        from tools.ganesh_tool import risk_prediction_model_tool
        g = await risk_prediction_model_tool(PATIENT)
        assert g["status"] in ("unavailable", "not_implemented", "ok")
        print(f"[OK] Risk Prediction Model: status={g['status']}"
              + (f"  reason: {g.get('reason','')}" if g["status"] != "ok" else ""))
    except Exception as e:
        errors.append(f"risk_prediction_model_tool: {e}")
        print(f"[FAIL] risk_prediction_model_tool: {e}")

    # 5. Prioritization Model (remote — Outreach Optimization)
    try:
        from tools.prediction_tool import prioritization_model_tool
        p = await prioritization_model_tool(PATIENT)
        assert p["status"] in ("unavailable", "not_implemented", "ok")
        print(f"[OK] Prioritization Model: status={p['status']}"
              + (f"  reason: {p.get('reason','')}" if p["status"] != "ok" else ""))
    except Exception as e:
        errors.append(f"prioritization_model_tool: {e}")
        print(f"[FAIL] prioritization_model_tool: {e}")

    # 5b. Star Impact Model (remote — Medicare Advantage Simulator)
    try:
        from tools.star_impact_tool import star_impact_model_tool
        si = await star_impact_model_tool(PATIENT)
        assert si["status"] in ("unavailable", "not_implemented", "ok")
        print(f"[OK] Star Impact Model: status={si['status']}"
              + (f"  reason: {si.get('reason','')}" if si["status"] != "ok" else ""))
    except Exception as e:
        errors.append(f"star_impact_model_tool: {e}")
        print(f"[FAIL] star_impact_model_tool: {e}")

    # 6. Health check import
    try:
        from agent.health_check import check_remote_services
        print("[OK] health_check imported")
    except Exception as e:
        errors.append(f"health_check: {e}")
        print(f"[FAIL] health_check: {e}")

    # 7. Orchestrator import
    try:
        from agent.orchestrator import run_analysis, TOOL_REGISTRY
        assert "risk_score_analysis"   in TOOL_REGISTRY
        assert "risk_prediction_model" in TOOL_REGISTRY
        assert "prioritization_model"  in TOOL_REGISTRY
        assert "star_impact_model"     in TOOL_REGISTRY
        print(f"[OK] Orchestrator imported, registry keys: {list(TOOL_REGISTRY.keys())}")
    except Exception as e:
        errors.append(f"orchestrator: {e}")
        print(f"[FAIL] orchestrator: {e}")

    # 8. FastAPI app import
    try:
        from main import app
        routes = [r.path for r in app.routes]
        assert "/analyze/{patient_id}" in routes
        assert "/health" in routes
        print(f"[OK] FastAPI app imported, routes: {routes}")
    except Exception as e:
        errors.append(f"main/fastapi: {e}")
        print(f"[FAIL] main/fastapi: {e}")

    print()
    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("All checks passed.")

asyncio.run(main())
