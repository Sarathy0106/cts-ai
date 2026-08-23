# Tools implemented for the AI Priority Agent
# Uses 5-component weighted scoring:
# Gap Impact (30%), Closure Feasibility (25%), Urgency (20%), Disease Severity (15%), Outreach Responsiveness (10%)

DISEASE_SEVERITY_MAP = {
    "diabetes mellitus": 0.85,
    "type 2 diabetes": 0.80,
    "hypertension": 0.70,
    "multiple chronic conditions": 0.95,
    "post-discharge": 0.90,
    "polypharmacy": 0.75,
    "colorectal screening eligible": 0.40,
    "breast cancer screening eligible": 0.40,
}

MEASURE_GAP_IMPACT = {
    "DM-01": 0.90,
    "MA-01": 0.90,
    "MA-02": 0.88,
    "MA-03": 0.85,
    "BP-01": 0.75,
    "CB-01": 0.70,
    "CB-02": 0.68,
    "FU-01": 0.80,
}


def score_member_priority(member, dataset_evidence):
    """
    Calculates a weighted priority score for a member.
    Components:
        - gap_impact (30%): derived from the measure code's star weight
        - closure_feasibility (25%): from risk prediction probability (care_gap_probability)
        - urgency (20%): based on days_since_last_contact and discharge status
        - disease_severity (15%): based on condition burden
        - outreach_responsiveness (10%): from K-Means segment and contact history
    """
    member_id = member.get("id", member.get("member_id", ""))
    risk_score = float(member.get("aiScore", member.get("risk_score", 0.75)))
    conditions = [c.lower() for c in (member.get("conditions") or [])]
    open_gaps = int(member.get("openGapsCount", member.get("open_gaps_count", 1)))
    priority_label = (member.get("priorityLabel") or member.get("priority_label") or "").lower()
    segment = (member.get("persona") or "").lower()
    contact_history = member.get("contactHistory") or []
    
    top_gap_code = ""
    if member.get("gaps"):
        for g in member["gaps"]:
            if g.get("status") == "Open":
                top_gap_code = g.get("measureCode", "")
                break

    # 1. Gap Impact (30%)
    gap_impact = MEASURE_GAP_IMPACT.get(top_gap_code, 0.60)
    # Bonus for multiple open gaps
    gap_impact = min(1.0, gap_impact + (open_gaps - 1) * 0.05)

    # 2. Closure Feasibility (25%) — from ML risk_score (care_gap_probability)
    closure_feasibility = min(1.0, risk_score)

    # 3. Urgency (20%) — higher if no recent contact, or post-discharge
    last_contact_days = 999
    if contact_history:
        from datetime import datetime
        try:
            last_date_str = contact_history[0].get("date", "")
            if last_date_str:
                delta = (datetime.utcnow() - datetime.fromisoformat(last_date_str)).days
                last_contact_days = delta
        except Exception:
            pass

    if "post-discharge" in conditions or last_contact_days <= 30:
        urgency_score = 0.95
    elif last_contact_days <= 60:
        urgency_score = 0.75
    elif last_contact_days <= 120:
        urgency_score = 0.55
    else:
        urgency_score = 0.40

    # 4. Disease Severity (15%)
    severity_scores = [DISEASE_SEVERITY_MAP.get(c, 0.50) for c in conditions]
    disease_severity = max(severity_scores) if severity_scores else 0.50

    # 5. Outreach Responsiveness (10%)
    if "high-utilization" in segment:
        outreach_resp = 0.80
    elif "routine" in segment:
        outreach_resp = 0.65
    else:
        outreach_resp = 0.40

    # Try to find from evidence if member_id is in dataset
    for record in dataset_evidence:
        raw = record.get("metadata", {}).get("raw_data", {})
        if raw.get("member_id", "").strip() == member_id:
            closure_feasibility = float(raw.get("closure_feasibility", closure_feasibility))
            outreach_resp = float(raw.get("outreach_responsiveness", outreach_resp))
            disease_severity = float(raw.get("disease_burden_score", disease_severity))
            break

    # Weighted composite score
    score = (
        gap_impact * 0.30 +
        closure_feasibility * 0.25 +
        urgency_score * 0.20 +
        disease_severity * 0.15 +
        outreach_resp * 0.10
    )
    score = round(min(100, score * 100), 1)

    return {
        "member_id": member_id,
        "priority_score": score,
        "components": {
            "gap_impact": round(gap_impact, 3),
            "closure_feasibility": round(closure_feasibility, 3),
            "urgency": round(urgency_score, 3),
            "disease_severity": round(disease_severity, 3),
            "outreach_responsiveness": round(outreach_resp, 3)
        },
        "top_gap_code": top_gap_code,
        "last_contact_days": last_contact_days,
        "requires_human_review": False
    }


def classify_urgency(priority_score, priority_label, last_contact_days, conditions):
    """
    Classifies the member's urgency tier for scheduling purposes.
    """
    label = priority_label.lower()
    has_discharge = any("post-discharge" in c.lower() for c in (conditions or []))
    
    if priority_score >= 85 or has_discharge or (label == "critical" and last_contact_days > 30):
        return {"urgency": "Immediate", "requires_escalation": True}
    elif priority_score >= 70 or label in ("critical", "high"):
        return {"urgency": "Standard", "requires_escalation": False}
    else:
        return {"urgency": "Routine", "requires_escalation": False}


def select_segment_strategy(segment, dataset_evidence):
    """
    Selects outreach strategy and care intensity based on K-Means segment.
    Grounded in segment_strategy.csv RAG evidence.
    """
    segment_lower = segment.lower()

    # Scan dataset evidence for segment match
    for record in dataset_evidence:
        raw = record.get("metadata", {}).get("raw_data", {})
        record_segment = raw.get("segment", "").lower()
        if record_segment and record_segment in segment_lower:
            return {
                "segment_strategy": raw.get("recommended_outreach_frequency", "Bi-weekly"),
                "preferred_channels": raw.get("preferred_channels", "Phone"),
                "care_intensity": raw.get("care_intensity", "Moderate"),
                "resource_allocation": raw.get("resource_allocation", "35%"),
                "source": record,
                "requires_human_review": False
            }

    # Fallback mapping
    if "high-utilization" in segment_lower or "complex" in segment_lower:
        return {
            "segment_strategy": "Weekly",
            "preferred_channels": "Phone;Portal",
            "care_intensity": "High",
            "resource_allocation": "40%",
            "source": None,
            "requires_human_review": False
        }
    elif "routine" in segment_lower:
        return {
            "segment_strategy": "Bi-weekly",
            "preferred_channels": "Phone;SMS",
            "care_intensity": "Moderate",
            "resource_allocation": "35%",
            "source": None,
            "requires_human_review": False
        }
    else:
        return {
            "segment_strategy": "Monthly",
            "preferred_channels": "SMS;Email",
            "care_intensity": "Low",
            "resource_allocation": "25%",
            "source": None,
            "requires_human_review": True
        }


def rank_members(scored_members):
    """
    Sorts members by priority_score descending and assigns rank positions.
    """
    ranked = sorted(scored_members, key=lambda m: m.get("priority_score", 0), reverse=True)
    for i, m in enumerate(ranked):
        m["rank"] = i + 1
    return ranked
