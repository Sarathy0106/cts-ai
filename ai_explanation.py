import pandas as pd
from ollama import chat

# Load datasets
members = pd.read_csv("processed_members.csv")
plans = pd.read_csv("plans.csv")


def calculate_score(member, plan, budget):
    score = 0

    # Premium affordability
    if plan["monthly_premium"] <= budget:
        score += 30
    elif plan["monthly_premium"] <= budget * 1.10:
        score += 15

    # Healthcare expense level
    if member["HEALTHCARE_EXPENSES"] >= 15000:
        if plan["deductible"] <= 3000:
            score += 20
        elif plan["deductible"] <= 5000:
            score += 10
    else:
        if plan["monthly_premium"] <= 300:
            score += 20

    # Coverage
    if plan["hospitalization"] == "Yes":
        score += 15

    if plan["prescription"] == "Yes":
        score += 15

    if plan["mental_health"] == "Yes":
        score += 10

    return score


def get_recommendations(member_id, budget=400):

    member_rows = members[
        members["Id"] == member_id
    ]

    if member_rows.empty:
        raise ValueError("Member not found.")

    member = member_rows.iloc[0]

    results = []

    for _, plan in plans.iterrows():

        score = calculate_score(
            member,
            plan,
            budget
        )

        results.append({
            "plan_id": plan["plan_id"],
            "plan_name": plan["plan_name"],
            "plan_type": plan["plan_type"],
            "monthly_premium": plan["monthly_premium"],
            "deductible": plan["deductible"],
            "doctor_copay": plan["doctor_copay"],
            "coinsurance": plan["coinsurance"],
            "hospitalization": plan["hospitalization"],
            "prescription": plan["prescription"],
            "mental_health": plan["mental_health"],
            "dental": plan["dental"],
            "vision": plan["vision"],
            "out_of_pocket_max": plan["out_of_pocket_max"],
            "network_size": plan["network_size"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return member, results[:3]


def generate_ai_explanation(member, recommendations):

    prompt = f"""
You are an insurance recommendation assistant.

Your job is to explain a recommendation that has ALREADY
been calculated by a recommendation engine.

IMPORTANT RULES:
- Do not invent insurance companies.
- Do not invent prices.
- Do not invent benefits.
- Do not invent medical conditions.
- Use ONLY the information provided below.
- Do not change the ranking.
- Do not make up information that is not present.

MEMBER INFORMATION:

Member ID: {member["Id"]}
Age: {member["AGE"]}
Location: {member["CITY"]}, {member["STATE"]}
Historical healthcare expenses: ${member["HEALTHCARE_EXPENSES"]}
Historical healthcare coverage: ${member["HEALTHCARE_COVERAGE"]}

TOP RECOMMENDED PLANS:

"""

    for index, plan in enumerate(recommendations, start=1):

        prompt += f"""
Rank {index}:
Plan: {plan["plan_name"]}
Type: {plan["plan_type"]}
Monthly Premium: ${plan["monthly_premium"]}
Deductible: ${plan["deductible"]}
Doctor Copay: ${plan["doctor_copay"]}
Coinsurance: {plan["coinsurance"]}
Hospitalization: {plan["hospitalization"]}
Prescription: {plan["prescription"]}
Mental Health: {plan["mental_health"]}
Dental: {plan["dental"]}
Vision: {plan["vision"]}
Out-of-Pocket Maximum: ${plan["out_of_pocket_max"]}
Network Size: {plan["network_size"]}
Recommendation Score: {plan["score"]}

"""

    prompt += """
Now provide:

1. The best recommended plan.
2. Why it received the highest score.
3. A comparison with the other two plans.
4. Important cost considerations.
5. A short final recommendation.

Keep the explanation clear and easy for a customer to understand.
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

    member_id = "P003"
    budget = 400

    member, recommendations = get_recommendations(
        member_id,
        budget
    )

    print("\n================================")
    print("TOP 3 RECOMMENDATIONS")
    print("================================")

    for i, plan in enumerate(recommendations, start=1):

        print(
            f"{i}. {plan['plan_name']} "
            f"→ Score {plan['score']}"
        )

    print("\n================================")
    print("LLAMA 3.1 8B EXPLANATION")
    print("================================")

    explanation = generate_ai_explanation(
        member,
        recommendations
    )

    print(explanation)