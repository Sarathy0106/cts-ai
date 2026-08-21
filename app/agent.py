import os
import json
import logging
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from prompts import AGENT_SYSTEM_PROMPT
from rag import retrieve_evidence
import tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIImpactAgent")


def run_agent(plan_data):
    """
    Executes the Impact Agent which assesses the star rating and financial
    impact of closing care gaps across the plan's member population.

    Args:
        plan_data (dict): {
            "plan_id": str,
            "current_stars": float,
            "measures": [
                {
                    "measure_code": str,
                    "measure_name": str,
                    "eligible": int,
                    "completed": int,
                    "gaps": int,
                    "gaps_to_close": int   # how many gaps we're targeting
                }
            ]
        }

    Returns:
        dict: Structured impact assessment plan
    """
    plan_id = plan_data.get("plan_id", "PLAN-001")
    current_stars = plan_data.get("current_stars", 3.5)
    measures_input = plan_data.get("measures", [])

    logger.info(f"[Impact Agent] Starting impact analysis for plan: {plan_id} (current stars: {current_stars})")

    # 1. Build RAG query from plan measure context
    measure_names = [m.get("measure_code", "") for m in measures_input]
    rag_query = f"Medicare Advantage star rating impact {' '.join(measure_names)} gap closure financial bonus QBP"
    logger.info(f"[Impact Agent] RAG query: '{rag_query}'")

    # 2. Retrieve evidence
    doc_evidence, data_evidence = retrieve_evidence(rag_query, top_k_docs=3, top_k_data=6)
    logger.info(f"[Impact Agent] Retrieved {len(doc_evidence)} docs and {len(data_evidence)} dataset records.")

    # 3. Call tools to assess impact on each measure
    measure_impacts = []
    total_gaps_closed = 0

    for m in measures_input:
        code = m.get("measure_code", "")
        eligible = int(m.get("eligible", 1000))
        completed = int(m.get("completed", 700))
        gaps_to_close = int(m.get("gaps_to_close", m.get("gaps", 0)))

        impact = tools.assess_measure_impact(
            measure_code=code,
            gaps_closed=gaps_to_close,
            eligible=eligible,
            completed=completed,
            dataset_evidence=data_evidence
        )
        measure_impacts.append(impact)
        total_gaps_closed += gaps_to_close
        logger.info(f"[Impact Agent] measure_impact for {code}: proj={impact['projected_performance_pct']}% star_contrib={impact['star_contribution']}")

    # 4. Project star rating
    star_projection = tools.project_star_rating(current_stars, measure_impacts)
    projected_stars = star_projection["projected_stars"]
    star_change = star_projection["star_change"]
    logger.info(f"[Impact Agent] Star projection: {current_stars} -> {projected_stars} (change: {star_change})")

    # 5. Calculate financial impact
    financial = tools.calculate_financial_impact(current_stars, projected_stars, total_gaps_closed, data_evidence)
    logger.info(f"[Impact Agent] Financial: bonus_change=${financial['projected_bonus'] - financial['current_bonus']:,}, net_gain=${financial['net_gain']:,}, ROI={financial['roi_pct']}%")

    # 6. Rank measures by ROI
    ranked = tools.rank_measures_by_roi(measure_impacts)

    # 7. Collect evidence sources
    evidence_sources = []
    for impact in measure_impacts:
        src = impact.get("source")
        if src:
            entry = {
                "source_type": src["source_type"],
                "source_name": src["source_name"],
                "reference": str(src.get("metadata", {}).get("row_index", ""))
            }
            if entry not in evidence_sources:
                evidence_sources.append(entry)

    # 8. Check human review flag
    requires_review = any(m.get("requires_human_review") for m in measure_impacts) or star_projection.get("requires_human_review", False)

    # 9. Determine top opportunity measure
    top_measure = ranked[0]["measure_code"] if ranked else ""

    # 10. Call LLM for final structured formatting
    prompt = f"""
    Plan ID: {plan_id}
    Current Star Rating: {current_stars}
    Projected Star Rating: {projected_stars}
    Star Change: {star_change}

    Measure Impact Summary:
    {json.dumps([{
        'measure_code': m['measure_code'],
        'current_pct': m['current_performance_pct'],
        'projected_pct': m['projected_performance_pct'],
        'star_contribution': m['star_contribution'],
        'financial_impact': m.get('financial_impact', 0),
        'priority_rank': m.get('priority_rank', 0)
    } for m in ranked], indent=2)}

    Financial Impact:
    {json.dumps(financial, indent=2)}

    Top Opportunity Measure: {top_measure}
    Requires Human Review: {requires_review}

    RAG Dataset Evidence:
    {json.dumps([{'name': d['source_name'], 'content': d['content'][:250]} for d in data_evidence], indent=2)}

    Format the response as a valid JSON object matching the requested schema exactly.
    """

    logger.info("[Impact Agent] Calling LLM for final structured formatting...")

    llm_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": AGENT_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=llm_payload, timeout=300)
        response.raise_for_status()
        llm_text = response.json().get("response", "").strip()

        if "```json" in llm_text:
            llm_text = llm_text.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_text:
            llm_text = llm_text.split("```")[1].split("```")[0].strip()

        final_plan = json.loads(llm_text)
        logger.info("[Impact Agent] LLM returned structured plan successfully.")
        return final_plan

    except Exception as e:
        logger.error(f"[Impact Agent] LLM error: {e}. Returning tool-built structured plan.")

    # Fallback: return tool-built structured plan
    return {
        "plan_id": plan_id,
        "current_stars": current_stars,
        "projected_stars": projected_stars,
        "star_change": star_change,
        "measure_impacts": [
            {
                "measure_code": m["measure_code"],
                "measure_name": m["measure_name"],
                "gaps_closed": measures_input[i].get("gaps_to_close", 0) if i < len(measures_input) else 0,
                "current_performance_pct": m["current_performance_pct"],
                "projected_performance_pct": m["projected_performance_pct"],
                "star_contribution": m["star_contribution"],
                "priority_rank": m.get("priority_rank", i + 1)
            }
            for i, m in enumerate(ranked)
        ],
        "financial_impact": {
            "current_bonus": financial["current_bonus"],
            "projected_bonus": financial["projected_bonus"],
            "net_gain": financial["net_gain"],
            "roi_pct": financial["roi_pct"]
        },
        "top_opportunity_measure": top_measure,
        "evidence": evidence_sources,
        "requires_human_review": requires_review
    }
