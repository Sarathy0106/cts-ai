import pandas as pd


# =========================================================
# LOAD COVERAGE RULES
# =========================================================

coverage_rules = pd.read_csv("coverage_rules.csv")


# =========================================================
# DETERMINE AGE GROUP
# =========================================================

def get_age_group(age):

    if age < 35:
        return "18-34"

    elif age < 50:
        return "35-49"

    elif age < 65:
        return "50-64"

    else:
        return "65+"


# =========================================================
# ANALYZE DEMOGRAPHIC
# =========================================================

def analyze_demographic(age):

    age_group = get_age_group(age)

    rules = coverage_rules[
        coverage_rules["age_group"] == age_group
    ].copy()

    return age_group, rules


# =========================================================
# RECOMMEND COVERAGES
# =========================================================

def recommend_coverages(age):

    age_group, rules = analyze_demographic(age)

    high_priority = rules[
        rules["priority"] == "High"
    ]

    medium_priority = rules[
        rules["priority"] == "Medium"
    ]

    low_priority = rules[
        rules["priority"] == "Low"
    ]

    return {
        "age_group": age_group,
        "high": high_priority.to_dict("records"),
        "medium": medium_priority.to_dict("records"),
        "low": low_priority.to_dict("records")
    }


# =========================================================
# EVALUATE PROPOSED PLAN
# =========================================================

def evaluate_plan(age, plan):

    recommendation = recommend_coverages(age)

    gaps = []
    covered = []

    # ---------------------------------------------
    # HIGH PRIORITY
    # ---------------------------------------------

    for item in recommendation["high"]:

        coverage = item["coverage"]

        column_name = coverage.lower().replace(" ", "_")

        if column_name in plan:

            value = str(
                plan[column_name]
            ).lower()

            if value == "yes":

                covered.append(coverage)

            else:

                gaps.append({
                    "coverage": coverage,
                    "priority": "High",
                    "reason": item["reason"]
                })


    # ---------------------------------------------
    # MEDIUM PRIORITY
    # ---------------------------------------------

    for item in recommendation["medium"]:

        coverage = item["coverage"]

        column_name = coverage.lower().replace(" ", "_")

        if column_name in plan:

            value = str(
                plan[column_name]
            ).lower()

            if value == "yes":

                covered.append(coverage)

            else:

                gaps.append({
                    "coverage": coverage,
                    "priority": "Medium",
                    "reason": item["reason"]
                })


    # ---------------------------------------------
    # SUITABILITY SCORE
    # ---------------------------------------------

    total_high = len(
        recommendation["high"]
    )

    high_gaps = len([
        gap
        for gap in gaps
        if gap["priority"] == "High"
    ])


    if total_high > 0:

        suitability = (
            (total_high - high_gaps)
            / total_high
        ) * 100

    else:

        suitability = 0


    return {
        "age_group": recommendation["age_group"],
        "covered": covered,
        "gaps": gaps,
        "suitability": round(
            suitability,
            2
        )
    }


# =========================================================
# TEST THE SYSTEM
# =========================================================

if __name__ == "__main__":

    plan = {

        "hospital_care": "Yes",

        "primary_care": "Yes",

        "specialist_care": "Yes",

        "preventive_care": "Yes",

        "prescription_drugs": "Yes",

        "emergency_care": "Yes",

        "diagnostic_services": "Yes",

        "mental_health": "Yes",

        "rehabilitation": "No",

        "dental": "No",

        "vision": "No"
    }


    result = evaluate_plan(
        65,
        plan
    )


    print("\n================================")
    print("PLAN EVALUATION")
    print("================================")


    print(
        "Target Age Group:",
        result["age_group"]
    )


    print(
        "Suitability:",
        result["suitability"],
        "%"
    )


    print("\nCovered:")


    for item in result["covered"]:

        print(
            "✓",
            item
        )


    print("\nCoverage Gaps:")


    for gap in result["gaps"]:

        print(
            "⚠",
            gap["coverage"],
            "(",
            gap["priority"],
            ")"
        )