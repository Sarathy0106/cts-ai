COVERAGE_WEIGHTS = {

    "hospital_care": 10,
    "primary_care": 10,
    "specialist_care": 10,
    "preventive_care": 10,
    "prescription_drugs": 15,
    "emergency_care": 10,
    "diagnostic_services": 10,
    "mental_health": 5,
    "rehabilitation": 10,
    "dental": 5,
    "vision": 5
}


def calculate_score(plan, coverage_type):

    score = 0
    total = 0

    for coverage, weight in COVERAGE_WEIGHTS.items():

        total += weight

        value = str(plan[coverage]).lower()

        if value == "yes":

            # Part D shouldn't receive points
            # for medical coverage it doesn't provide.

            if coverage_type == "Part D":

                if coverage == "prescription_drugs":
                    score += weight

            else:
                score += weight

    return round(
        (score / total) * 100,
        2
    )


def rating_from_score(score):

    if score >= 90:
        return 5

    elif score >= 80:
        return 4

    elif score >= 70:
        return 3

    elif score >= 60:
        return 2

    else:
        return 1