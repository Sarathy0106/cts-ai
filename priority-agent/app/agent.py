import os
import json
import logging
import requests
from datetime import datetime
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, SEGMENTATION_API_URL
from prompts import AGENT_SYSTEM_PROMPT
from rag import retrieve_evidence
import tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIPriorityAgent")

# Priority label mapping
SCORE_LABEL_MAP = [
    (85, "Critical"),
    (70, "High"),
    (55, "Medium"),
    (0, "Low"),
]


def _score_to_label(score):
    for threshold, label in SCORE_LABEL_MAP:
        if score >= threshold:
            return label
    return "Low"


def _get_segment_from_api(member):
    """
    Calls the K-Means segmentation model API for the member.
    Falls back to persona field or rule-based classification on failure.
    """
    if member.get("persona"):
        return member["persona"]
    
    try:
        priority_lower = (member.get("priorityLabel") or "").lower()
        if "critical" in priority_lower or "high" in priority_lower:
            encounters, outpatient, inpatient, emergency, providers, days_since = 12.0, 8.0, 2.0, 2.0, 4.0, 15.0
        elif "medium" in priority_lower:
            encounters, outpatient, inpatient, emergency, providers, days_since = 5.0, 4.0, 0.0, 1.0, 2.0, 45.0
        else:
            encounters, outpatient, inpatient, emergency, providers, days_since = 1.0, 1.0, 0.0, 0.0, 1.0, 180.0

        payload = {
            "member_id": str(member.get("id") or member.get("member_id", "MBR-100234")),
            "encounter_count_12m": float(member.get("encounter_count_12m", encounters)),
            "outpatient_visits_12m": float(member.get("outpatient_visits_12m", outpatient)),
            "inpatient_visits_12m": float(member.get("inpatient_visits_12m", inpatient)),
            "emergency_visits_12m": float(member.get("emergency_visits_12m", emergency)),
            "unique_provider_count_12m": float(member.get("unique_provider_count_12m", providers)),
            "days_since_last_encounter": float(member.get("days_since_last_encounter", days_since))
        }
        resp = requests.post(f"{SEGMENTATION_API_URL}/predict", json=payload, timeout=5)
        if resp.ok:
            seg_data = resp.json()
            return seg_data.get("persona") or seg_data.get("segment_label") or seg_data.get("segment", "Routine / Moderate Utilization")
    except Exception as e:
        logger.warning(f"[Priority Agent] Segmentation API failed: {e}. Using rule-based fallback.")

    # Rule-based fallback
    risk = float(member.get("aiScore", 0.7))
    gaps = int(member.get("openGapsCount", 1))
    if risk >= 0.80 or gaps >= 2:
        return "High-Utilization / Complex-Care Pattern"
    elif risk >= 0.60:
        return "Routine / Moderate Utilization"
    else:
        return "Low-Utilization / Inactive Care Pattern"


def run_agent(members_data):
    """
    Executes the Priority Agent which ranks members by care intervention urgency.

    Args:
        members_data (dict): {
            "plan_id": str,
            "members": [ ... list of member dicts from frontend/backend ... ]
        }

    Returns:
        dict: Structured priority queue with ranked members and resource allocation
    """
    plan_id = members_data.get("plan_id", "PLAN-001")
    members = members_data.get("members", [])

    logger.info(f"[Priority Agent] Starting priority ranking for {len(members)} members in plan: {plan_id}")

    # 1. Build RAG query
    rag_query = "member priority care gap urgency disease severity outreach responsiveness segment strategy"
    doc_evidence, data_evidence = retrieve_evidence(rag_query, top_k_docs=3, top_k_data=5)
    logger.info(f"[Priority Agent] Retrieved {len(doc_evidence)} docs and {len(data_evidence)} data records.")

    # 2. Score and process each member
    scored_members = []
    all_evidence_sources = []

    for member in members:
        member_id = member.get("id") or member.get("member_id", "")
        name = member.get("name", "Unknown")
        conditions = member.get("conditions") or []

        # 2a. Get segment from API or fallback
        segment = _get_segment_from_api(member)
        logger.info(f"[Priority Agent] Member {member_id} -> segment: '{segment}'")

        # 2b. Score priority
        score_result = tools.score_member_priority(member, data_evidence)
        priority_score = score_result["priority_score"]
        priority_label = _score_to_label(priority_score)
        last_contact_days = score_result["last_contact_days"]
        top_gap_code = score_result["top_gap_code"]

        # 2c. Classify urgency
        urgency_result = tools.classify_urgency(priority_score, priority_label, last_contact_days, conditions)

        # 2d. Select segment strategy
        strategy = tools.select_segment_strategy(segment, data_evidence)

        # 2e. Build reasons list
        reasons = []
        if priority_score >= 85:
            reasons.append("Critical risk score with multiple or high-impact open care gaps")
        if last_contact_days < 30:
            reasons.append(f"Recent contact {last_contact_days} days ago — within 30-day follow-up window")
        if any("post-discharge" in c.lower() for c in conditions):
            reasons.append("Post-discharge patient requiring follow-up within 30 days per FU-01 measure")
        if len(conditions) >= 2:
            reasons.append(f"Multiple chronic conditions ({', '.join(conditions[:3])})")
        if top_gap_code in ("MA-01", "MA-02", "MA-03"):
            reasons.append(f"Medication adherence gap ({top_gap_code}) — triple-weighted CMS measure")
        if not reasons:
            reasons.append("Open care gap with clinical risk indicators detected")

        # Collect evidence sources
        src = strategy.get("source")
        if src:
            entry = {
                "source_type": src["source_type"],
                "source_name": src["source_name"],
                "reference": str(src.get("metadata", {}).get("row_index", ""))
            }
            if entry not in all_evidence_sources:
                all_evidence_sources.append(entry)

        scored_members.append({
            "member_id": member_id,
            "name": name,
            "priority_score": priority_score,
            "priority_label": priority_label,
            "urgency": urgency_result["urgency"],
            "requires_escalation": urgency_result["requires_escalation"],
            "segment": segment,
            "segment_strategy": strategy["segment_strategy"],
            "recommended_channel": strategy["preferred_channels"].split(";")[0] if strategy["preferred_channels"] else "Phone",
            "care_intensity": strategy["care_intensity"],
            "top_gap": top_gap_code,
            "reasons": reasons,
            "components": score_result["components"]
        })

        logger.info(f"[Priority Agent] {member_id}: score={priority_score} | label={priority_label} | urgency={urgency_result['urgency']}")

    # 3. Rank all members
    ranked = tools.rank_members(scored_members)

    # 4. Calculate resource allocation
    high_count = sum(1 for m in ranked if m["care_intensity"] == "High")
    moderate_count = sum(1 for m in ranked if m["care_intensity"] == "Moderate")
    low_count = sum(1 for m in ranked if m["care_intensity"] == "Low")

    requires_review = any(m.get("requires_escalation") for m in ranked)
    timestamp = datetime.utcnow().isoformat()

    # 5. Call LLM for final structured validation
    prompt = f"""
    Plan ID: {plan_id}
    Generated At: {timestamp}
    Total Members: {len(ranked)}

    Priority Queue Summary:
    {json.dumps([{
        'member_id': m['member_id'],
        'name': m['name'],
        'priority_score': m['priority_score'],
        'priority_label': m['priority_label'],
        'urgency': m['urgency'],
        'segment': m['segment'],
        'care_intensity': m['care_intensity'],
        'top_gap': m['top_gap'],
        'requires_escalation': m['requires_escalation'],
        'reasons': m['reasons']
    } for m in ranked[:10]], indent=2)}

    Resource Allocation:
    High Intensity: {high_count}, Moderate: {moderate_count}, Low: {low_count}

    RAG Evidence Retrieved:
    {json.dumps([{'name': d['source_name'], 'content': d['content'][:200]} for d in data_evidence], indent=2)}

    Format the response as a valid JSON object matching the requested schema exactly.
    """

    logger.info("[Priority Agent] Calling LLM for final structured validation...")

    llm_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": AGENT_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=llm_payload, timeout=60)
        response.raise_for_status()
        llm_text = response.json().get("response", "").strip()

        if "```json" in llm_text:
            llm_text = llm_text.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_text:
            llm_text = llm_text.split("```")[1].split("```")[0].strip()

        final_plan = json.loads(llm_text)
        logger.info("[Priority Agent] LLM returned structured plan successfully.")
        return final_plan

    except Exception as e:
        logger.error(f"[Priority Agent] LLM error: {e}. Returning tool-built structured plan.")

    # Fallback: tool-built structured plan
    return {
        "plan_id": plan_id,
        "generated_at": timestamp,
        "total_members": len(ranked),
        "priority_queue": [
            {
                "member_id": m["member_id"],
                "name": m["name"],
                "priority_score": m["priority_score"],
                "priority_label": m["priority_label"],
                "urgency": m["urgency"],
                "segment": m["segment"],
                "segment_strategy": m["segment_strategy"],
                "recommended_channel": m["recommended_channel"],
                "care_intensity": m["care_intensity"],
                "top_gap": m["top_gap"],
                "reasons": m["reasons"],
                "requires_escalation": m["requires_escalation"]
            }
            for m in ranked
        ],
        "resource_allocation": {
            "high_intensity_count": high_count,
            "moderate_intensity_count": moderate_count,
            "low_intensity_count": low_count
        },
        "evidence": all_evidence_sources,
        "requires_human_review": requires_review
    }
