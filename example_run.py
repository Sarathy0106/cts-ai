"""
End-to-end example: Patient Input → full pipeline → Quality Gap Analysis JSON

Shows:
  1. The context object assembled from all tool outputs
  2. The exact messages sent to Ollama
  3. The raw Ollama response
  4. The final parsed JSON

Run with:
    python example_run.py

Requires Ollama to be running (ollama serve) with llama3.1:8b pulled.
"""
from __future__ import annotations
import asyncio
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Test patient ───────────────────────────────────────────────────────────────
PATIENT_ID = "P-00142"
PATIENT_DATA = {
    "age": 68, "sex": "female",
    "conditions": ["diabetes", "hypertension"],
    "medications_count": 6,
    "ed_visits_90d": 1,
    "hospitalizations_180d": 0,
    "has_pcp": True,
    "med_adherence_pct": 55.0,
    "smoker": False,
    "bmi": 31.2,
    "last_a1c_date":                "2023-01-15",
    "last_a1c_value":               9.4,
    "last_bp_systolic":             148,
    "last_bp_diastolic":            92,
    "last_mammogram_date":          "2021-06-10",
    "last_flu_vaccine_date":        "2022-09-01",
    "last_kidney_eval_date":        None,
    "last_wellness_visit_date":     None,
    "last_med_reconciliation_date": None,
    "last_colonoscopy_date":        None,
    "last_depression_screen_date":  None,
    "last_pap_date":                "2024-01-01",
}


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


async def main():
    # ── Pre-flight: Ollama reachability ───────────────────────────────────────
    from agent import ollama_client
    section("Pre-flight: Ollama check")
    reachable = await ollama_client.is_ollama_reachable()
    if not reachable:
        print("ERROR: Ollama is NOT reachable at http://localhost:11434")
        print()
        print("To start Ollama:    ollama serve")
        print("To pull the model:  ollama pull llama3.1:8b")
        sys.exit(1)

    models = await ollama_client.list_models()
    print(f"Ollama is running. Available models: {models}")
    if ollama_client.DEFAULT_MODEL not in models:
        print(f"\nWARNING: '{ollama_client.DEFAULT_MODEL}' is not pulled.")
        print(f"Run:  ollama pull {ollama_client.DEFAULT_MODEL}")
        sys.exit(1)
    print(f"Model '{ollama_client.DEFAULT_MODEL}' confirmed available.")

    # ── Step A: Run all tools and build context ───────────────────────────────
    section("Step A: Tool outputs + context assembly")
    from agent.context_builder import build_context
    from agent.reasoning_prompt import build_messages
    from tools.risk_score_tool import risk_score_tool
    from tools.care_gap_tool import care_gap_tool
    from tools.segmentation_tool import segmentation_tool
    from tools.ganesh_tool import risk_prediction_model_tool
    from tools.prediction_tool import prioritization_model_tool
    from tools.star_impact_tool import star_impact_model_tool

    risk_result, care_gap_result = await asyncio.gather(
        risk_score_tool(PATIENT_DATA),
        care_gap_tool(PATIENT_DATA),
    )
    segment_result = await segmentation_tool(
        PATIENT_DATA,
        risk_score   = risk_result.get("score") or 0,
        care_gap_count = care_gap_result.get("gap_count") or 0,
    )
    risk_pred_result, priority_result, star_result = await asyncio.gather(
        risk_prediction_model_tool(PATIENT_DATA),
        prioritization_model_tool(PATIENT_DATA),
        star_impact_model_tool(PATIENT_DATA),
    )

    context = build_context(
        patient_id        = PATIENT_ID,
        patient_data      = PATIENT_DATA,
        risk_result       = risk_result,
        care_gap_result   = care_gap_result,
        segment_result    = segment_result,
        risk_pred_result  = risk_pred_result,
        priority_result   = priority_result,
        star_result       = star_result,
    )

    print("\n[Context object sent to reasoning prompt]")
    print(json.dumps(context, indent=2, default=str))

    # ── Step B: Build messages ────────────────────────────────────────────────
    section("Step B: Ollama chat messages")
    messages = build_messages(context)
    for i, m in enumerate(messages):
        print(f"\n--- message[{i}] role={m['role']} ({len(m['content'])} chars) ---")
        # Print first 800 chars of each message to keep output readable
        print(m["content"][:800])
        if len(m["content"]) > 800:
            print(f"... [{len(m['content'])-800} more chars]")

    # ── Step C: Call Ollama ───────────────────────────────────────────────────
    section("Step C: Raw Ollama response")
    print(f"Calling {ollama_client.DEFAULT_MODEL} (this may take 30–90 s)...")
    try:
        raw_content, raw_resp = await ollama_client.chat(messages=messages)
    except ollama_client.OllamaError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"\n[Raw response — {len(raw_content)} chars]")
    print(raw_content)

    # ── Step D: Parse ─────────────────────────────────────────────────────────
    section("Step D: Parsed final JSON")
    try:
        parsed = ollama_client.parse_json_response(raw_content)
        # Apply the same post-processing the orchestrator does
        parsed.setdefault("patient_id",         PATIENT_ID)
        parsed.setdefault("models_unavailable",  context["models_unavailable"])
        parsed.setdefault("models_used",         context["models_available"])
        parsed.setdefault("derived_facts",       context["derived_facts"])
        if not isinstance(parsed.get("raw_ml_predictions"), dict):
            from agent.orchestrator import _raw_predictions_from_context
            parsed["raw_ml_predictions"] = _raw_predictions_from_context(context)
        print(json.dumps(parsed, indent=2, default=str))
    except Exception as exc:
        print(f"JSON parse failed: {exc}")
        print("Raw LLM output was:")
        print(raw_content)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    unavail = parsed.get("models_unavailable", [])
    print(f"Patient ID         : {parsed.get('patient_id')}")
    print(f"Models used        : {parsed.get('models_used')}")
    print(f"Models unavailable : {[m['model'] for m in unavail]}")
    raw_preds = parsed.get("raw_ml_predictions", {})
    rs = raw_preds.get("risk_score_analysis", {})
    cg = raw_preds.get("care_gap_analysis", {})
    sg = raw_preds.get("patient_segmentation", {})
    print(f"Risk Score         : {rs.get('score')} ({rs.get('risk_level')})")
    print(f"Care Gaps          : {cg.get('gap_count')} found, highest={cg.get('highest_severity')}")
    print(f"Segment            : {sg.get('segment_id')} — {sg.get('segment_label')}")
    print(f"Confidence         : {parsed.get('analysis_confidence')}")
    if parsed.get("confidence_note"):
        print(f"Confidence note    : {parsed['confidence_note']}")
    recs = parsed.get("recommendations", [])
    print(f"Recommendations    : {len(recs)}")
    for r in recs:
        print(f"  [{r.get('urgency','?').upper()}] {r.get('action','')}")


if __name__ == "__main__":
    asyncio.run(main())
