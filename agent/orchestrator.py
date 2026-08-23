"""
Quality Gap Analysis Agent — Orchestrator

Flow per /analyze/{patient_id} request:

  Patient Input
       ↓
  1. Health-check remote services (parallel TCP, 3 s timeout each)
       ↓
  2. Run local tools in parallel
     ┌─────────────────┬──────────────────┐
     │ Risk Score Tool │ Care Gap Tool     │
     └────────┬────────┴────────┬─────────┘
              └────────┬────────┘
                       ↓  (both results in hand)
              Segmentation Tool  (needs risk_score + gap_count)
       ↓
  3. Run 3 remote tools in parallel
     ┌──────────────────────┬─────────────────────┬────────────────────┐
     │ Risk Prediction Tool │ Prioritization Tool │ Star Impact Tool   │
     └──────────────────────┴─────────────────────┴────────────────────┘
       ↓
  4. Assemble combined context object  (context_builder.build_context)
       ↓
  5. Build Ollama chat messages         (reasoning_prompt.build_messages)
       ↓
  6. Call Ollama /api/chat              (ollama_client.chat)
       ↓
  7. Parse + validate JSON response     (ollama_client.parse_json_response)
       ↓
  Final Quality Gap Analysis JSON  →  returned to /analyze/{patient_id}
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any

from agent.health_check import check_remote_services
from agent.context_builder import build_context
from agent.reasoning_prompt import build_messages
from agent import ollama_client
from tools.risk_score_tool import risk_score_tool
from tools.care_gap_tool import care_gap_tool
from tools.segmentation_tool import segmentation_tool
from tools.ganesh_tool import risk_prediction_model_tool
from tools.prediction_tool import prioritization_model_tool
from tools.star_impact_tool import star_impact_model_tool

logger = logging.getLogger(__name__)

# ── Tool registry ──────────────────────────────────────────────────────────────
# Extend here — orchestration logic does not need to change.
TOOL_REGISTRY = {
    "risk_score_analysis":    risk_score_tool,
    "care_gap_analysis":      care_gap_tool,
    "patient_segmentation":   segmentation_tool,  # called after risk + gap (needs their outputs)
    "risk_prediction_model":  risk_prediction_model_tool,
    "prioritization_model":   prioritization_model_tool,
    "star_impact_model":      star_impact_model_tool,
}

OLLAMA_MODEL       = ollama_client.DEFAULT_MODEL   # "llama3.1:8b"
OLLAMA_TEMPERATURE = 0.1                           # low temp for structured clinical output


async def run_analysis(patient_id: str, patient_data: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point — called by POST /analyze/{patient_id}.
    Returns the full Quality Gap Analysis as a dict.
    Never raises — all errors are captured into structured fallback responses.
    """
    logger.info("=== Starting analysis: patient_id=%s ===", patient_id)

    # ── Step 1: Health-check remote services ──────────────────────────────────
    logger.info("Step 1 — Health-checking remote ML services...")
    service_statuses = await check_remote_services()
    for s in service_statuses:
        logger.info("  %-30s %s -> %s",
                    s.name, s.url, "UP" if s.reachable else f"DOWN ({s.detail})")

    # ── Step 2: Local tools (risk + care gap in parallel, then segmentation) ──
    logger.info("Step 2 — Running local ML tools...")
    risk_result, care_gap_result = await asyncio.gather(
        risk_score_tool(patient_data),
        care_gap_tool(patient_data),
    )
    logger.info("  Risk Score:  score=%s  level=%s  status=%s",
                risk_result.get("score"), risk_result.get("risk_level"), risk_result.get("status"))
    logger.info("  Care Gaps:   count=%s  highest=%s  status=%s",
                care_gap_result.get("gap_count"), care_gap_result.get("highest_severity"),
                care_gap_result.get("status"))

    risk_score_val = risk_result.get("score") or 0
    gap_count_val  = care_gap_result.get("gap_count") or 0
    segment_result = await segmentation_tool(patient_data, risk_score_val, gap_count_val)
    logger.info("  Segmentation: segment=%s  status=%s",
                segment_result.get("segment_id"), segment_result.get("status"))

    # -- Step 3: Remote tools (3 new services) ------------------------------------
    logger.info("Step 3 -- Running remote ML tools...")
    risk_pred_result, priority_result, star_result = await asyncio.gather(
        risk_prediction_model_tool(patient_data),
        prioritization_model_tool(patient_data),
        star_impact_model_tool(patient_data),
    )
    logger.info("  Risk Prediction Model: status=%s", risk_pred_result.get("status"))
    logger.info("  Prioritization Model:  status=%s", priority_result.get("status"))
    logger.info("  Star Impact Model:     status=%s", star_result.get("status"))

    # -- Step 4: Assemble context -----------------------------------------------
    logger.info("Step 4 -- Assembling combined ML context...")
    context = build_context(
        patient_id              = patient_id,
        patient_data            = patient_data,
        risk_result             = risk_result,
        care_gap_result         = care_gap_result,
        segment_result          = segment_result,
        risk_pred_result        = risk_pred_result,
        priority_result         = priority_result,
        star_result             = star_result,
    )
    logger.info("  models_available=%s  models_unavailable=%s",
                context["models_available"],
                [m["model"] for m in context["models_unavailable"]])

    # ── Step 5: Build Ollama messages ─────────────────────────────────────────
    logger.info("Step 5 — Building Ollama chat messages...")
    messages = build_messages(context)
    total_chars = sum(len(m["content"]) for m in messages)
    logger.info("  %d messages, %d total chars sent to Ollama", len(messages), total_chars)

    # ── Step 6: Call Ollama ───────────────────────────────────────────────────
    logger.info("Step 6 — Calling Ollama (model=%s)...", OLLAMA_MODEL)
    raw_content, raw_ollama_resp = await _call_ollama_safe(messages)
    if raw_content is None:
        # Ollama unreachable or hard error — raw_ollama_resp holds the error string
        logger.warning("Ollama call failed — returning fallback with local model outputs only")
        return _build_fallback(context, error=str(raw_ollama_resp))

    # ── Step 7: Parse JSON response ───────────────────────────────────────────
    logger.info("Step 7 — Parsing Ollama JSON response (%d chars)...", len(raw_content))
    return _parse_and_finalise(raw_content, context)


async def _call_ollama_safe(
    messages: list[dict[str, str]],
) -> tuple[str | None, Any]:
    """
    Wraps ollama_client.chat — returns (content, raw_resp) on success,
    or (None, error_string) on failure.  Never raises.
    """
    try:
        content, raw = await ollama_client.chat(
            messages    = messages,
            model       = OLLAMA_MODEL,
            temperature = OLLAMA_TEMPERATURE,
        )
        return content, raw
    except ollama_client.OllamaError as exc:
        logger.error("OllamaError: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.error("Unexpected error calling Ollama: %s", exc, exc_info=True)
        return None, str(exc)


def _parse_and_finalise(raw_content: str, context: dict[str, Any]) -> dict[str, Any]:
    """
    Attempts JSON extraction from the LLM response.
    Falls back gracefully if parsing fails.
    """
    try:
        parsed = ollama_client.parse_json_response(raw_content)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (%s) — returning fallback. Raw: %.200s", exc, raw_content)
        return _build_fallback(
            context,
            error=f"LLM response JSON parse failed: {exc}",
            raw_llm_output=raw_content,
        )

    # Guarantee these keys are always present regardless of what the LLM did
    patient_id = context["patient_summary"]["patient_id"]
    parsed.setdefault("patient_id",         patient_id)
    parsed.setdefault("models_unavailable", context["models_unavailable"])
    parsed.setdefault("models_used",        context["models_available"])
    parsed.setdefault("derived_facts",      context["derived_facts"])

    # Inject the pre-built raw_ml_predictions if LLM omitted them or garbled them
    if not isinstance(parsed.get("raw_ml_predictions"), dict):
        parsed["raw_ml_predictions"] = _raw_predictions_from_context(context)

    logger.info("Analysis complete — confidence=%s  unavailable=%d",
                parsed.get("analysis_confidence", "?"),
                len(parsed.get("models_unavailable", [])))
    return parsed


def _build_fallback(
    context: dict[str, Any],
    error: str = "",
    raw_llm_output: str = "",
) -> dict[str, Any]:
    """
    Returned when Ollama is unavailable or returns unparseable output.
    Local model outputs are always preserved.
    """
    patient_id = context["patient_summary"]["patient_id"]
    result = {
        "patient_id":         patient_id,
        "models_used":        context["models_available"],
        "models_unavailable": context["models_unavailable"],
        "raw_ml_predictions": _raw_predictions_from_context(context),
        "derived_facts":      context["derived_facts"],
        "agent_reasoning": {
            "error": error or "Ollama reasoning unavailable",
            "note":  "Local model outputs are valid. Re-run when Ollama is available.",
        },
        "recommendations":     [],
        "analysis_confidence": "low",
        "confidence_note":     "LLM reasoning step failed — see agent_reasoning.error",
    }
    if raw_llm_output:
        result["_debug_raw_llm_output"] = raw_llm_output[:2000]
    return result


def _raw_predictions_from_context(context: dict[str, Any]) -> dict[str, Any]:
    t = context["tool_outputs"]
    def _val(key: str) -> Any:
        r = t[key]
        if r.get("status") in ("unavailable", "error", "not_implemented"):
            return f"unavailable -- {r.get('reason', 'no reason returned')}"
        return r
    return {
        "risk_score_analysis":   _val("risk_score_analysis"),
        "care_gap_analysis":     _val("care_gap_analysis"),
        "patient_segmentation":  _val("patient_segmentation"),
        "risk_prediction_model": _val("risk_prediction_model"),
        "prioritization_model":  _val("prioritization_model"),
        "star_impact_model":     _val("star_impact_model"),
    }
