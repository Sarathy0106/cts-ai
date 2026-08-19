import pandas as pd


COVERAGE_COLUMNS = [
    "hospital_care",
    "primary_care",
    "specialist_care",
    "preventive_care",
    "prescription_drugs",
    "emergency_care",
    "diagnostic_services",
    "mental_health",
    "rehabilitation",
    "dental",
    "vision"
]


def analyze_plan(plan):

    gaps = []
    strengths = []

    for coverage in COVERAGE_COLUMNS:

        value = str(plan[coverage]).lower()

        readable = coverage.replace("_", " ").title()

        if value == "yes":
            strengths.append(readable)

        else:
            gaps.append(readable)

    return {
        "strengths": strengths,
        "gaps": gaps
    }


def load_plans():

    return pd.read_csv("plans.csv")


def get_plan(company, plan_name):

    df = load_plans()

    result = df[
        (df["company"] == company) &
        (df["plan_name"] == plan_name)
    ]

    if result.empty:
        return None

    return result.iloc[0]