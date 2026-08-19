import pandas as pd


# Load datasets
members = pd.read_csv("processed_members.csv")
plans = pd.read_csv("plans.csv")


def calculate_score(member, plan, budget):
    score = 0

    # -------------------------------------------------
    # 1. Premium affordability
    # -------------------------------------------------
    if plan["monthly_premium"] <= budget:
        score += 30
    elif plan["monthly_premium"] <= budget * 1.10:
        score += 15

    # -------------------------------------------------
    # 2. Healthcare expense level
    # -------------------------------------------------
    if member["HEALTHCARE_EXPENSES"] >= 15000:

        # High healthcare expenses -> prefer lower deductible
        if plan["deductible"] <= 3000:
            score += 20
        elif plan["deductible"] <= 5000:
            score += 10

    else:

        # Lower healthcare expenses -> prioritize lower premium
        if plan["monthly_premium"] <= 300:
            score += 20

    # -------------------------------------------------
    # 3. Hospitalization
    # -------------------------------------------------
    if plan["hospitalization"] == "Yes":
        score += 15

    # -------------------------------------------------
    # 4. Prescription
    # -------------------------------------------------
    if plan["prescription"] == "Yes":
        score += 15

    # -------------------------------------------------
    # 5. Mental health
    # -------------------------------------------------
    if plan["mental_health"] == "Yes":
        score += 10

    return score


def recommend(member_id, budget=400):

    # Find member
    member_rows = members[
        members["Id"] == member_id
    ]

    if member_rows.empty:
        print("Member not found.")
        return

    member = member_rows.iloc[0]

    results = []

    # Compare member against every plan
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
            "score": score
        })

    # Sort highest score first
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # Display results
    print("\n================================")
    print("MEMBER PROFILE")
    print("================================")

    print("Member ID:", member["Id"])
    print("Age:", member["AGE"])
    print("Location:", member["CITY"], ",", member["STATE"])
    print(
        "Healthcare Expenses:",
        member["HEALTHCARE_EXPENSES"]
    )

    print("\n================================")
    print("PLAN RANKING")
    print("================================")

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. {result['plan_name']} "
            f"({result['plan_type']})"
        )

        print(
            f"   Premium: ${result['monthly_premium']}/month"
        )

        print(
            f"   Deductible: ${result['deductible']}"
        )

        print(
            f"   Score: {result['score']}"
        )

    print("\n================================")
    print("TOP 3 RECOMMENDATIONS")
    print("================================")

    for result in results[:3]:

        print(
            f"{result['plan_name']} "
            f"→ Score {result['score']}"
        )

    return member, results[:3]


# Test the recommendation system
if __name__ == "__main__":

    recommend(
        member_id="P003",
        budget=400
    )