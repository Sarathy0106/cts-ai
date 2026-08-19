import streamlit as st

from plan_analyzer import (
    load_plans,
    get_plan,
    analyze_plan
)

from plan_scorer import (
    calculate_score,
    rating_from_score
)

from plan_optimizer import (
    optimize_plan
)

from plan_ai import (
    generate_optimization_explanation
)


st.set_page_config(
    page_title="Insurance Plan Optimizer",
    page_icon="🏥",
    layout="wide"
)


st.title("🏥 AI Insurance Plan Optimizer")

st.write(
    "White-label platform for analyzing and improving "
    "insurance products."
)


# =====================================================
# LOAD DATA
# =====================================================

df = load_plans()


# =====================================================
# COMPANY
# =====================================================

st.header("🏢 Insurance Company")

company = st.selectbox(
    "Select Insurance Company",
    sorted(df["company"].unique())
)


# =====================================================
# PLAN
# =====================================================

company_plans = df[
    df["company"] == company
]

plan_name = st.selectbox(
    "Select Existing Plan",
    company_plans["plan_name"].tolist()
)


plan = get_plan(
    company,
    plan_name
)


# =====================================================
# DISPLAY CURRENT PLAN
# =====================================================

st.header("📋 Current Plan")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Target Age",
        plan["target_age"]
    )

with col2:

    st.metric(
        "Coverage",
        plan["coverage_type"]
    )

with col3:

    st.metric(
        "Current Rating",
        f"⭐ {plan['current_rating']}"
    )


# =====================================================
# ANALYZE
# =====================================================

if st.button(
    "🔍 Analyze & Improve Plan",
    use_container_width=True
):

    analysis = analyze_plan(plan)

    current_score = calculate_score(
        plan,
        plan["coverage_type"]
    )

    current_rating = rating_from_score(
        current_score
    )

    # =================================================
    # CURRENT ANALYSIS
    # =================================================

    st.divider()

    st.header("🔎 Current Plan Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Existing Strengths")

        for item in analysis["strengths"]:

            st.write(
                f"✓ {item}"
            )

    with col2:

        st.subheader("⚠️ Coverage Gaps")

        for item in analysis["gaps"]:

            st.write(
                f"⚠ {item}"
            )


    # =================================================
    # OPTIMIZATION
    # =================================================

    optimization = optimize_plan(
        plan,
        plan["coverage_type"]
    )


    st.divider()

    st.header("🚀 Recommended Improvements")

    if optimization["improvements"]:

        for item in optimization["improvements"]:

            st.write(
                f"➕ Add / strengthen {item}"
            )

    else:

        st.success(
            "No major coverage improvements detected."
        )


    # =================================================
    # BEFORE / AFTER
    # =================================================

    st.divider()

    st.header("📊 Before vs After")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Current Plan"
        )

        st.metric(
            "Estimated Score",
            f"{current_score}%"
        )

        st.write(
            "⭐" * current_rating
        )


    with col2:

        st.subheader(
            "Improved Plan"
        )

        st.metric(
            "Estimated Score",
            f"{optimization['score']}%",
            delta=f"{optimization['score'] - current_score:.1f}"
        )

        st.write(
            "⭐" * optimization["rating"]
        )


    # =================================================
    # IMPROVED PLAN
    # =================================================

    st.divider()

    st.header("🏆 AI-Optimized Plan")

    improved = optimization["plan"]

    st.subheader(
        f"{plan_name} — Improved"
    )

    st.write(
        f"Target: {improved['target_age']}"
    )

    st.write(
        f"Coverage: {improved['coverage_type']}"
    )

    st.write(
        f"Estimated Rating: "
        f"{'⭐' * optimization['rating']} "
        f"{optimization['rating']}/5"
    )


    st.subheader(
        "Coverage"
    )

    for column in [
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
    ]:

        if str(
            improved[column]
        ).lower() == "yes":

            st.write(
                f"✓ {column.replace('_', ' ').title()}"
            )


    # =================================================
    # AI EXPLANATION
    # =================================================

    st.divider()

    st.header(
        "🤖 Llama 3.1 8B Product Insight"
    )

    with st.spinner(
        "Analyzing plan improvement..."
    ):

        explanation = (
            generate_optimization_explanation(
                company,
                plan,
                analysis,
                optimization
            )
        )

    st.write(explanation)