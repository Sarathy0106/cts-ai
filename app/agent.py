import os
import json
import logging
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL
from prompts import AGENT_SYSTEM_PROMPT
from rag import retrieve_evidence
import tools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIInterventionAgent")

def run_agent(ml_result):
    """
    Executes tool calling and reasoning over ML results and dual-source RAG context
    to produce the final structured JSON plan.
    """
    patient_id = ml_result.get("patient_id")
    risk_factors = ml_result.get("risk_factors", [])
    logger.info(f"Starting agent intervention analysis for patient: {patient_id}")

    # 1. Generate RAG query from risk factors and predictions
    query_parts = [ml_result.get("prediction", ""), ml_result.get("risk_level", "")] + risk_factors
    rag_query = " ".join([q for q in query_parts if q])
    logger.info(f"Retrieving RAG evidence for query: '{rag_query}'")

    # 2. Retrieve evidence
    doc_evidence, data_evidence = retrieve_evidence(rag_query, top_k_docs=3, top_k_data=3)
    logger.info(f"Retrieved {len(doc_evidence)} docs and {len(data_evidence)} dataset records.")

    # 3. Call tools to construct grounded answers
    logger.info("Executing clinical tools...")
    intervention_res = tools.select_intervention(ml_result, data_evidence, doc_evidence)
    logger.info(f"select_intervention output: {intervention_res}")

    channel_res = tools.choose_channel(data_evidence)
    logger.info(f"choose_channel output: {channel_res}")

    followup_res = tools.create_followup_plan(data_evidence)
    logger.info(f"create_followup_plan output: {followup_res}")

    outreach_msg = tools.generate_outreach(
        patient_id=patient_id,
        selected_intervention=intervention_res["intervention"],
        risk_factors=risk_factors
    )

    # 4. Consolidate evidence sources
    evidence_sources = []
    for res in [intervention_res, channel_res, followup_res]:
        src = res.get("source")
        if src:
            source_entry = {
                "source_type": src["source_type"],
                "source_name": src["source_name"],
                "reference": str(src.get("metadata", {}).get("page_number", "")) or str(src.get("metadata", {}).get("row_index", ""))
            }
            if source_entry not in evidence_sources:
                evidence_sources.append(source_entry)

    # 5. Check if any tool flagged human review
    requires_human_review = (
        intervention_res["requires_human_review"] or
        channel_res["requires_human_review"] or
        followup_res["requires_human_review"]
    )
    logger.info(f"Initial human review flag status: {requires_human_review}")

    # 6. Call LLM for final reasoning and validation alignment
    prompt = f"""
    Patient ID: {patient_id}
    ML Prediction: {ml_result}
    
    Selected Intervention: {intervention_res['intervention']}
    Intervention Reason: {intervention_res['reason']}
    Selected Channel: {channel_res['channel']}
    Outreach Message: {outreach_msg}
    Follow-up: Action: {followup_res['action']}, Timeframe: {followup_res['timeframe']}
    
    Retrieved Documentation context:
    {json.dumps([{ 'name': d['source_name'], 'text': d['content'][:300] } for d in doc_evidence])}
    
    Retrieved Dataset context:
    {json.dumps([{ 'name': d['source_name'], 'content': d['content'] } for d in data_evidence])}
    
    Human Review Required: {requires_human_review}
    
    Format the response as a valid JSON object matching the requested schema. Ensure all fields are filled accurately.
    """

    logger.info("Calling LLM for final validation and structured formatting...")
    
    llm_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": AGENT_SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=llm_payload, timeout=60)
        response.raise_for_status()
        llm_response_text = response.json().get("response", "").strip()
        
        # Locate the JSON block in case there is markdown wrapper
        if "```json" in llm_response_text:
            llm_response_text = llm_response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response_text:
            llm_response_text = llm_response_text.split("```")[1].split("```")[0].strip()

        # Parse LLM formatted plan
        final_plan = json.loads(llm_response_text)
        logger.info("Successfully received structured response from Ollama LLM.")
        return final_plan

    except Exception as e:
        logger.warning(f"Error querying Ollama or parsing response: {e}")
        
        if GROQ_API_KEY:
            logger.info("Attempting fallback to Groq API...")
            groq_payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            groq_headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            try:
                groq_response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=groq_payload,
                    headers=groq_headers,
                    timeout=30
                )
                groq_response.raise_for_status()
                groq_json = groq_response.json()
                llm_response_text = groq_json["choices"][0]["message"]["content"].strip()
                
                # Locate the JSON block in case there is markdown wrapper
                if "```json" in llm_response_text:
                    llm_response_text = llm_response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_response_text:
                    llm_response_text = llm_response_text.split("```")[1].split("```")[0].strip()

                final_plan = json.loads(llm_response_text)
                logger.info("Successfully received structured response from Groq LLM.")
                return final_plan
            except Exception as groq_e:
                logger.error(f"Groq API fallback failed: {groq_e}")

        # Fallback to local structured plan if both LLMs fail or GROQ_API_KEY is not set
        logger.warning("Falling back to local template response.")
        return {
            "patient_id": patient_id,
            "ml_assessment": ml_result,
            "intervention_plan": {
                "intervention": intervention_res["intervention"],
                "reason": intervention_res["reason"],
                "channel": channel_res["channel"],
                "outreach_message": outreach_msg,
                "follow_up": {
                    "action": followup_res["action"],
                    "timeframe": followup_res["timeframe"]
                }
            },
            "evidence": evidence_sources,
            "requires_human_review": True
        }
