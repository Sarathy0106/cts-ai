from ollama import chat


def generate_optimization_explanation(
    company,
    plan,
    analysis,
    optimization
):

    prompt = f"""
You are an AI insurance product optimization advisor.

Insurance company:
{company}

Existing plan:
{plan['plan_name']}

Target age:
{plan['target_age']}

Coverage type:
{plan['coverage_type']}

Current rating:
{plan['current_rating']}/5

Current strengths:
{analysis['strengths']}

Current gaps:
{analysis['gaps']}

Recommended improvements:
{optimization['improvements']}

Estimated improved score:
{optimization['score']}/100

Estimated improved rating:
{optimization['rating']}/5

Explain:

1. Why the existing plan may be underperforming.
2. Which coverage gaps were identified.
3. Why each recommended improvement may strengthen
   the product for the target demographic.
4. What the improved plan looks like.
5. Give a concise product-manager recommendation.

IMPORTANT:

- Do not claim this is an official CMS Star Rating.
- Do not invent regulatory approval.
- Do not invent prices.
- Do not invent statistics.
- Do not claim the improved rating is guaranteed.
- Call it an estimated/model score.
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