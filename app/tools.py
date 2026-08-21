# Tools implemented for the AI Intervention Agent

def select_intervention(ml_result, dataset_evidence, documentation_evidence):
    """
    Selects an intervention grounded in RAG evidence matching patient risk factors.
    """
    risk_factors = ml_result.get("risk_factors", [])

    # Scan datasets for a match with any of the risk factors
    for record in dataset_evidence:
        content = record.get("content", "").lower()
        raw_data = record.get("metadata", {}).get("raw_data", {})

        # Check if the record specifies an intervention
        for factor in risk_factors:
            clean_factor = factor.split("(")[0].strip().lower()
            if clean_factor in content or raw_data.get("risk_factor", "").lower() in clean_factor:
                intervention = raw_data.get("intervention") or raw_data.get("successful_intervention")
                recommendation = raw_data.get("recommendation")
                if intervention:
                    return {
                        "intervention": intervention,
                        "reason": f"Matched risk factor '{factor}' with clinical guideline mapping to: {recommendation or intervention}",
                        "requires_human_review": False,
                        "source": record
                    }

    # Scan documentation for a match
    for doc in documentation_evidence:
        content = doc.get("content", "").lower()
        for factor in risk_factors:
            clean_factor = factor.split("(")[0].strip().lower()
            if clean_factor in content:
                # Basic text extraction match
                return {
                    "intervention": f"Guideline-driven care coordination for {factor}",
                    "reason": f"Extracted clinical recommendation matching '{factor}' from documentation page {doc.get('metadata', {}).get('page_number')}",
                    "requires_human_review": False,
                    "source": doc
                }

    return {
        "intervention": "Generic care outreach",
        "reason": "No matching clinical intervention mapped in RAG evidence. Care manager review required.",
        "requires_human_review": True,
        "source": None
    }


def generate_outreach(patient_id, selected_intervention, risk_factors):
    """
    Creates outreach copy tailored to the patient and selected intervention.
    """
    if not selected_intervention or selected_intervention == "Generic care outreach":
        return "Dear patient, we would like to schedule a routine wellness check-in with your primary care provider."

    factors_str = " and ".join([f.split("(")[0].strip() for f in risk_factors])
    message = f"Hello, this is your Care Management Team. We noticed some indicators relating to {factors_str}. To support you, we would like to coordinate a '{selected_intervention}' to check on your health."
    return message


def choose_channel(dataset_evidence):
    """
    Selects a communication channel supported by dataset evidence.
    Does not use fallback/assumptions unless explicitly matching evidence.
    """
    for record in dataset_evidence:
        raw_data = record.get("metadata", {}).get("raw_data", {})
        channel = raw_data.get("channel") or raw_data.get("successful_channel")
        if channel:
            return {
                "channel": channel,
                "requires_human_review": False,
                "source": record
            }

    # Return review if no evidence is found
    return {
        "channel": "REQUIRES_HUMAN_REVIEW",
        "requires_human_review": True,
        "source": None
    }


def create_followup_plan(dataset_evidence):
    """
    Selects follow-up timeframe and action from dataset evidence.
    """
    for record in dataset_evidence:
        raw_data = record.get("metadata", {}).get("raw_data", {})
        follow_up = raw_data.get("follow_up") or raw_data.get("follow_up_action")
        timeframe = raw_data.get("timeframe")

        # Determine timeframe if it's in the follow_up text
        if follow_up:
            if not timeframe:
                # Simple extraction of days/weeks from the string if present
                import re
                match = re.search(r'\b\d+\s+(?:days|weeks|months)\b', follow_up, re.IGNORECASE)
                timeframe = match.group(0) if match else "as specified"

            return {
                "action": follow_up,
                "timeframe": timeframe,
                "requires_human_review": False,
                "source": record
            }

    return {
        "action": "Verify patient status and outreach response",
        "timeframe": "REQUIRES_HUMAN_REVIEW",
        "requires_human_review": True,
        "source": None
    }
