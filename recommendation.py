import pandas as pd
from ollama import chat


# Load data
members = pd.read_csv("processed_members.csv")
plans = pd.read_csv("plans.csv")


def calculate_plan_score(member, plan, budget):
    score = 0

    # Premium affordability
    if plan["monthly_premium"] <= budget:
        score += 30
    elif plan["monthly_premium"] <= budget * 1.10:
        score += 15

    # Healthcare expense level
    if member["HEALTHCARE_EXPENSES"] > 10000:
        if plan["deductible"] <= 3000:
            score += 20
    else:
        if plan["monthly_premium"] <= 300:
            score += 20

    # Hospital coverage
    if plan["hospitalization"] == "Yes":
        score += 15

    # Prescription coverage
    if plan["prescription"] == "Yes":
        score += 15

    # Mental health coverage
    if plan["mental_health"] == "Yes":
        score += 10

    return score


def recommend(member_id, budget=400):

    member = members[
        members["Id"] == member_id
    ].iloc[0]

    results = []

    for _, plan in plans.iterrows():

        score = calculate_plan_score(
            member,
            plan,
            budget
        )

        results.append({
            "plan": plan["plan_name"],
            "type": plan["plan_type"],
            "premium": plan["monthly_premium"],
            "deductible": plan["deductible"],
            "score": score
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return member, results[:3]


def generate_explanation(member, recommendations):

    prompt = f"""
You are an insurance recommendation assistant.

You must explain the recommendations using ONLY the information
provided below.

Member information:
Age: {member["AGE"]}
State: {member["STATE"]}
County: {member["COUNTY"]}
Historical healthcare expenses: {member["HEALTHCARE_EXPENSES"]}
Historical healthcare coverage: {member["HEALTHCARE_COVERAGE"]}

Top recommended plans:

{recommendations}

Explain:

1. Which plan is the best match.
2. Why it matches the member.
3. Why the other plans are alternatives.
4. Mention important cost differences.

Do not invent:
- insurance companies
- prices
- benefits
- coverage
- medical conditions

Clearly state that this is a recommendation based on
the provided dataset.
"""

    response = chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

    if __name__ == "__main__":

    member_id = members.iloc[0]["Id"]

    member, recommendations = recommend(
        member_id,
        budget=400
    )

    print("\nTOP RECOMMENDATIONS")
    print("====================")

    for recommendation in recommendations:
        print(recommendation)

    print("\nAI EXPLANATION")
    print("====================")

    explanation = generate_explanation(
        member,
        recommendations
    )

    print(explanation)