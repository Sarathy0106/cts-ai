"""
Builds the Ollama chat messages from the assembled ML context.

Two messages are returned (system + user) so the orchestrator can pass them
directly to the /api/chat endpoint.

Design principles:
- System message is short and sets hard behavioural constraints.
- User message contains all data and the exact JSON schema to fill in.
- The schema is templated so the model only has to fill values, not invent structure.
- Unavailable models are called out explicitly — the model is forbidden to infer them.
"""
from __future__ import annotations
import json
from typing import Any

SYSTEM_MESSAGE = """\
You are a clinical quality gap analysis agent.
Your ONLY job is to fill in the JSON template provided by the user.
Rules you must never break:
1. Output ONLY the completed JSON object. No preamble. No explanation. No markdown fences.
2. Never fabricate data for any model listed under models_unavailable.
   For those models write exactly the string: "unavailable — <reason from context>"
3. Base every claim in agent_reasoning on data explicitly present in tool_outputs.
4. Do not add, rename, or remove any top-level JSON keys from the template.
5. urgency must be one of: immediate, soon, routine."""


def build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    """
    Returns [system_message, user_message] suitable for /api/chat messages[].
    """
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": _build_user_message(context)},
    ]


def _build_user_message(ctx: dict[str, Any]) -> str:
    def fmt(obj: Any) -> str:
        return json.dumps(obj, indent=2, default=str)

    patient   = ctx["patient_summary"]
    tools     = ctx["tool_outputs"]
    derived   = ctx["derived_facts"]
    unavail   = ctx["models_unavailable"]
    available = ctx["models_available"]

    # Build per-model data blocks dynamically from tool_outputs
    # unavailable ones get a clear notice; available ones show their full output
    def model_block(key: str, label: str) -> str:
        result = tools.get(key, {"status": "unavailable", "reason": "tool not registered"})
        if result.get("status") in ("unavailable", "error", "not_implemented"):
            reason = result.get("reason", "unknown")
            return (f"[{label}]\nSTATUS: {result['status']}\nREASON: {reason}\n"
                    f"DO NOT reason about this model's prediction.")
        return f"[{label}]\n{fmt(result)}"

    models_section = "\n\n".join([
        model_block("risk_score_analysis",   "Risk Score Analysis (local)"),
        model_block("care_gap_analysis",     "Care Gap Analysis (local)"),
        model_block("patient_segmentation",  "Patient Segmentation (local)"),
        model_block("risk_prediction_model", "Risk Prediction Model (remote — CMS Star Ratings)"),
        model_block("prioritization_model",  "Prioritization Model (remote — Outreach Optimization)"),
        model_block("star_impact_model",     "Star Impact Model (remote — Medicare Advantage Simulator)"),
    ])

    unavail_list = fmt(unavail) if unavail else "[]"
    derived_list = fmt(derived)

    # Build the JSON template — values in <angle brackets> are to be filled
    # Escape braces in the f-string so only our placeholders are live
    template = """{
  "patient_id": \"""" + patient["patient_id"] + """\",
  "models_used": """ + json.dumps(available) + """,
  "models_unavailable": """ + unavail_list + """,
  "raw_ml_predictions": {
    "risk_score_analysis":   <copy the full risk_score_analysis tool output verbatim>,
    "care_gap_analysis":     <copy the full care_gap_analysis tool output verbatim>,
    "patient_segmentation":  <copy the full patient_segmentation tool output verbatim>,
    "risk_prediction_model": <if unavailable write "unavailable -- <reason>", else copy verbatim>,
    "prioritization_model":  <if unavailable write "unavailable -- <reason>", else copy verbatim>,
    "star_impact_model":     <if unavailable write "unavailable -- <reason>", else copy verbatim>
  },
  "derived_facts": """ + derived_list + """,
  "agent_reasoning": {
    "quality_gaps_present":      "<list each gap id and severity from care_gap_analysis>",
    "most_significant_gap":      "<which gap is highest severity and why, citing care_gap_analysis>",
    "risk_level":                "<risk level and score from risk_score_analysis>",
    "patient_segment":           "<segment_id and label from patient_segmentation>",
    "model_consensus_summary":   "<what the AVAILABLE models collectively indicate — skip unavailable ones>",
    "cross_model_relationships": "<relationships between risk level, segment, gaps, and any remote predictions>",
    "priority_gap":              "<which gap to address first and why>",
    "supporting_evidence":       "<exact fields from tool outputs that support the priority>"
  },
  "recommendations": [
    {"action": "<specific action>", "rationale": "<why, citing a model output>", "urgency": "immediate|soon|routine"}
  ],
  "analysis_confidence": "<high if all 6 models ok, medium if 3-5 ok, low if <3 ok>",
  "confidence_note": "<note if any models unavailable, else omit or empty string>"
}"""

    return f"""=== PATIENT ===
{fmt(patient)}

=== MODEL OUTPUTS ===
{models_section}

=== DERIVED FACTS (pre-computed, do not change) ===
{derived_list}

=== MODELS UNAVAILABLE ===
{unavail_list}

=== INSTRUCTIONS ===
Fill in every <placeholder> in the JSON template below.
Return ONLY the completed JSON. Nothing before it. Nothing after it.

=== JSON TEMPLATE TO COMPLETE ===
{template}"""
