import copy

from plan_scorer import (
    calculate_score,
    rating_from_score
)


def optimize_plan(plan, coverage_type):

    improved_plan = copy.deepcopy(plan)

    required_coverages = [
        "hospital_care",
        "primary_care",
        "specialist_care",
        "preventive_care",
        "prescription_drugs",
        "emergency_care",
        "diagnostic_services",
        "mental_health",
        "rehabilitation"
    ]

    if coverage_type == "Part C":

        required_coverages.remove(
            "prescription_drugs"
        )

    elif coverage_type == "Part D":

        required_coverages = [
            "prescription_drugs"
        ]

    else:

        pass

    improvements = []

    for coverage in required_coverages:

        current = str(
            improved_plan[coverage]
        ).lower()

        if current != "yes":

            improved_plan[coverage] = "Yes"

            improvements.append(
                coverage.replace(
                    "_",
                    " "
                ).title()
            )

    new_score = calculate_score(
        improved_plan,
        coverage_type
    )

    new_rating = rating_from_score(
        new_score
    )

    return {
        "plan": improved_plan,
        "improvements": improvements,
        "score": new_score,
        "rating": new_rating
    }