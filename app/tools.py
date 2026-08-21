# Tools implemented for the AI Impact Agent

def assess_measure_impact(measure_code, gaps_closed, eligible, completed, dataset_evidence):
    """
    Scores the impact of closing gaps on a specific measure's performance
    and star rating contribution. Grounded in retrieved measure_impact.csv data.
    """
    for record in dataset_evidence:
        raw = record.get("metadata", {}).get("raw_data", {})
        if raw.get("measure_code", "").strip().upper() == measure_code.upper():
            weight = int(raw.get("weight", 1))
            cut_4star = float(raw.get("cms_cut_point_4star", 80))
            cut_5star = float(raw.get("cms_cut_point_5star", 90))
            financial_per_gap = float(raw.get("financial_impact_per_gap", 35))

            new_completed = completed + gaps_closed
            proj_pct = round((new_completed / eligible) * 100, 1) if eligible > 0 else 0.0
            curr_pct = round((completed / eligible) * 100, 1) if eligible > 0 else 0.0

            if proj_pct >= cut_5star:
                star_contribution = weight * 5
            elif proj_pct >= cut_4star:
                star_contribution = weight * 4
            else:
                star_contribution = weight * 3

            return {
                "measure_code": measure_code,
                "measure_name": raw.get("measure_name", measure_code),
                "weight": weight,
                "current_performance_pct": curr_pct,
                "projected_performance_pct": proj_pct,
                "star_contribution": star_contribution,
                "financial_impact": round(gaps_closed * financial_per_gap, 2),
                "star_impact_description": raw.get("star_impact_description", ""),
                "requires_human_review": False,
                "source": record
            }

    # Fallback if not found in evidence
    curr_pct = round((completed / eligible) * 100, 1) if eligible > 0 else 0.0
    proj_pct = round(((completed + gaps_closed) / eligible) * 100, 1) if eligible > 0 else 0.0
    return {
        "measure_code": measure_code,
        "measure_name": measure_code,
        "weight": 1,
        "current_performance_pct": curr_pct,
        "projected_performance_pct": proj_pct,
        "star_contribution": 3,
        "financial_impact": 0,
        "star_impact_description": "No RAG evidence found. Estimate only.",
        "requires_human_review": True,
        "source": None
    }


def project_star_rating(current_stars, measure_impacts):
    """
    Projects the plan's new star rating based on measure-level star contributions.
    Uses a weighted average approach matching CMS methodology.
    """
    total_weight = 0
    weighted_stars = 0
    
    for m in measure_impacts:
        w = m.get("weight", 1)
        stars = m.get("star_contribution", 3) / w  # normalized back to per-star
        weighted_stars += stars * w
        total_weight += w

    if total_weight == 0:
        return {"projected_stars": current_stars, "requires_human_review": True}

    avg_stars = weighted_stars / total_weight
    projected = round(avg_stars * 2) / 2.0  # Round to nearest 0.5
    projected = max(1.0, min(5.0, projected))
    star_change = round(projected - current_stars, 2)

    return {
        "projected_stars": projected,
        "star_change": star_change,
        "requires_human_review": False
    }


def calculate_financial_impact(current_stars, projected_stars, total_gaps_closed, dataset_evidence):
    """
    Calculates QBP bonus changes based on star rating tier transitions.
    Grounded in star_scenarios.csv.
    """
    # CMS QBP bonus approximations
    bonus_by_stars = {
        1.0: 0,
        1.5: 0,
        2.0: 0,
        2.5: 0,
        3.0: 1_500_000,
        3.5: 1_800_000,
        4.0: 3_800_000,
        4.5: 4_200_000,
        5.0: 5_000_000
    }

    current_bonus = bonus_by_stars.get(round(current_stars * 2) / 2, 1_800_000)
    projected_bonus = bonus_by_stars.get(round(projected_stars * 2) / 2, 1_800_000)

    # Try to pull closure cost from scenario evidence
    avg_cost_per_gap = 35.0
    for record in dataset_evidence:
        raw = record.get("metadata", {}).get("raw_data", {})
        if "closure_cost_per_member" in raw:
            avg_cost_per_gap = float(raw["closure_cost_per_member"])
            break

    intervention_cost = total_gaps_closed * avg_cost_per_gap
    net_gain = projected_bonus - current_bonus - intervention_cost
    roi = round((net_gain / intervention_cost) * 100, 1) if intervention_cost > 0 else 0.0

    return {
        "current_bonus": current_bonus,
        "projected_bonus": projected_bonus,
        "intervention_cost": round(intervention_cost, 2),
        "net_gain": round(net_gain, 2),
        "roi_pct": roi,
        "requires_human_review": False
    }


def rank_measures_by_roi(measure_impacts):
    """
    Ranks measures by their cost-effectiveness (financial impact per gap closed).
    """
    ranked = sorted(
        measure_impacts,
        key=lambda m: m.get("financial_impact", 0),
        reverse=True
    )
    for i, m in enumerate(ranked):
        m["priority_rank"] = i + 1

    return ranked
