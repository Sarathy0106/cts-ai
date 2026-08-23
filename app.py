"""
Quality Gap Analysis Agent — Streamlit frontend
================================================
IMPORTANT: The FastAPI backend must be running separately before using this app.
  Start the backend:  python -m uvicorn main:app --host 0.0.0.0 --port 8000
  Start this UI:      streamlit run app.py          (opens on port 8501)

This app is a single form-submit-report interface — not a chatbot.
It POSTs to http://localhost:8000/analyze/{patient_id} and renders the
structured JSON response in labelled sections.
"""
import json
from datetime import date, datetime

import pandas as pd
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE        = "http://localhost:8000"
ANALYZE_TIMEOUT = 180   # seconds — Ollama reasoning can be slow
HEALTH_TIMEOUT  = 8

# Conditions accepted by the API (union of care_gap measure checks + segmentation chronic_list)
ALL_CONDITIONS = [
    "diabetes",
    "hypertension",
    "heart_disease",
    "chf",
    "cad",
    "copd",
    "ckd",
    "chronic kidney disease",
    "cancer",
    "depression",
    "anxiety",
    "obesity",
]

# P-00142 defaults (same patient used in verify_local.py / example_run.py)
DEFAULTS = {
    "patient_id":               "P-00142",
    "age":                      68,
    "sex":                      "female",
    "conditions":               ["diabetes", "hypertension"],
    "medications_count":        6,
    "ed_visits_90d":            1,
    "hospitalizations_180d":    0,
    "has_pcp":                  True,
    "med_adherence_pct":        55.0,
    "smoker":                   False,
    "bmi":                      31.2,
    "last_a1c_value":           9.4,
    "last_a1c_date":            "2023-01-15",
    "last_bp_systolic":         148,
    "last_bp_diastolic":        92,
    "last_mammogram_date":      "2021-06-10",
    "last_flu_vaccine_date":    "2022-09-01",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "none": 4}

# ── Colour helpers ─────────────────────────────────────────────────────────────
def sev_badge(severity: str) -> str:
    colours = {
        "critical": "#c0392b", "high": "#e67e22",
        "moderate": "#f1c40f", "low": "#27ae60", "none": "#95a5a6",
    }
    text_col = "#fff" if severity in ("critical", "high") else "#222"
    bg = colours.get(severity.lower(), "#95a5a6")
    return (
        f'<span style="background:{bg};color:{text_col};padding:2px 10px;'
        f'border-radius:12px;font-size:0.82em;font-weight:600;'
        f'text-transform:uppercase;">{severity}</span>'
    )

def urgency_badge(urgency: str) -> str:
    colours = {"immediate": "#c0392b", "soon": "#e67e22", "routine": "#7f8c8d"}
    bg = colours.get(urgency.lower(), "#7f8c8d")
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 9px;'
        f'border-radius:10px;font-size:0.78em;font-weight:600;'
        f'text-transform:uppercase;">{urgency}</span>'
    )

def dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"

# ── API helpers ────────────────────────────────────────────────────────────────
def fetch_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as exc:
        return {"_error": str(exc)}

def run_analysis(patient_id: str, payload: dict) -> tuple[int, dict | str]:
    """Returns (http_status, parsed_json_or_error_string)."""
    try:
        r = requests.post(
            f"{API_BASE}/analyze/{patient_id}",
            json=payload,
            timeout=ANALYZE_TIMEOUT,
        )
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body
    except requests.exceptions.ConnectionError:
        return 0, "connection_refused"
    except requests.exceptions.Timeout:
        return 0, "timeout"
    except Exception as exc:
        return 0, str(exc)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quality Gap Analysis Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Health check banner ────────────────────────────────────────────────────────
def render_health_banner():
    health = fetch_health()
    if health is None:
        st.error(
            "🔴 **Backend not reachable.** Start it with:\n\n"
            "```\npython -m uvicorn main:app --host 0.0.0.0 --port 8000\n```"
        )
        return False

    if "_error" in health:
        st.warning(f"⚠️ Health check error: {health['_error']}")
        return True

    services = health.get("remote_services", [])
    lines = []
    all_remote_ok = True
    for s in services:
        ok = s.get("reachable", False)
        if not ok:
            all_remote_ok = False
        lines.append(f"{dot(ok)} **{s['name']}** ({s['host']}:{s['port']})")

    # Ollama: infer from whether the backend itself responded
    backend_line = f"{dot(True)} **FastAPI backend** (localhost:8000) — reachable"

    summary = "  |  ".join([backend_line] + lines)

    if all_remote_ok:
        st.success(f"🟢 All services reachable  ·  {summary}")
    else:
        st.warning(
            f"⚠️ Some remote ML services are unreachable (graceful degradation active)\n\n"
            + "  \n".join([backend_line] + lines)
        )
    return True

# ── Sidebar — patient input form ──────────────────────────────────────────────
def render_sidebar() -> tuple[str, dict] | tuple[None, None]:
    st.sidebar.title("🏥 Patient Input")
    st.sidebar.caption("Pre-filled with test patient P-00142. Edit any field and click Run Analysis.")

    with st.sidebar.form("patient_form"):
        st.subheader("Identity")
        patient_id = st.text_input("Patient ID", value=DEFAULTS["patient_id"])

        st.subheader("Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=130,
                                  value=DEFAULTS["age"], step=1)
        with col2:
            sex = st.selectbox("Sex", ["female", "male", "unknown"],
                               index=["female", "male", "unknown"].index(DEFAULTS["sex"]))

        st.subheader("Clinical Profile")
        conditions = st.multiselect(
            "Conditions",
            options=ALL_CONDITIONS,
            default=DEFAULTS["conditions"],
        )
        col3, col4 = st.columns(2)
        with col3:
            medications_count = st.number_input("Medications count", min_value=0,
                                                value=DEFAULTS["medications_count"], step=1)
            ed_visits_90d     = st.number_input("ED visits (last 90d)", min_value=0,
                                                value=DEFAULTS["ed_visits_90d"], step=1)
        with col4:
            hospitalizations  = st.number_input("Hospitalizations (last 180d)", min_value=0,
                                                value=DEFAULTS["hospitalizations_180d"], step=1)
            bmi               = st.number_input("BMI", min_value=10.0, max_value=80.0,
                                                value=float(DEFAULTS["bmi"]), step=0.1,
                                                format="%.1f")

        med_adherence = st.slider("Medication adherence (%)", 0, 100,
                                  value=int(DEFAULTS["med_adherence_pct"]))
        col5, col6 = st.columns(2)
        with col5:
            has_pcp = st.checkbox("Has PCP", value=DEFAULTS["has_pcp"])
        with col6:
            smoker  = st.checkbox("Smoker",  value=DEFAULTS["smoker"])

        st.subheader("Lab Values & Screening Dates")
        st.caption("Leave blank if unknown — missing values are treated as care gaps.")

        col7, col8 = st.columns(2)
        with col7:
            a1c_val = st.number_input("Last HbA1c value (%)", min_value=0.0, max_value=20.0,
                                      value=float(DEFAULTS["last_a1c_value"]),
                                      step=0.1, format="%.1f")
            bp_sys  = st.number_input("BP Systolic (mmHg)", min_value=0, max_value=300,
                                      value=DEFAULTS["last_bp_systolic"], step=1)
        with col8:
            a1c_date_raw = st.date_input("Last HbA1c test date",
                                         value=date.fromisoformat(DEFAULTS["last_a1c_date"]))
            bp_dia  = st.number_input("BP Diastolic (mmHg)", min_value=0, max_value=200,
                                      value=DEFAULTS["last_bp_diastolic"], step=1)

        col9, col10 = st.columns(2)
        with col9:
            mammo_raw = st.date_input("Last mammogram date",
                                      value=date.fromisoformat(DEFAULTS["last_mammogram_date"]))
        with col10:
            flu_raw   = st.date_input("Last flu vaccine date",
                                      value=date.fromisoformat(DEFAULTS["last_flu_vaccine_date"]))

        submitted = st.form_submit_button("🔍 Run Analysis", use_container_width=True,
                                          type="primary")

    if not submitted:
        return None, None

    payload = {
        "age":                   int(age),
        "sex":                   sex,
        "conditions":            conditions,
        "medications_count":     int(medications_count),
        "ed_visits_90d":         int(ed_visits_90d),
        "hospitalizations_180d": int(hospitalizations),
        "has_pcp":               has_pcp,
        "med_adherence_pct":     float(med_adherence),
        "smoker":                smoker,
        "bmi":                   float(bmi),
        "last_a1c_value":        float(a1c_val),
        "last_a1c_date":         a1c_date_raw.isoformat() if a1c_date_raw else None,
        "last_bp_systolic":      int(bp_sys) if bp_sys else None,
        "last_bp_diastolic":     int(bp_dia) if bp_dia else None,
        "last_mammogram_date":   mammo_raw.isoformat() if mammo_raw else None,
        "last_flu_vaccine_date": flu_raw.isoformat()   if flu_raw   else None,
    }
    return patient_id, payload

# ── Response renderers ────────────────────────────────────────────────────────

def render_header(result: dict):
    """Patient ID, risk badge, segment label."""
    raw = result.get("raw_ml_predictions", {})
    rs  = raw.get("risk_score_analysis", {})
    seg = raw.get("patient_segmentation", {})

    risk_level   = rs.get("risk_level", "unknown") if isinstance(rs, dict) else "unknown"
    risk_score   = rs.get("score", "?")            if isinstance(rs, dict) else "?"
    seg_label    = seg.get("segment_label", "?")   if isinstance(seg, dict) else "?"

    pid = result.get("patient_id", "?")
    st.markdown(
        f"## Patient `{pid}`  &nbsp; {sev_badge(risk_level)} &nbsp;"
        f"<span style='font-size:0.85em;color:#555;'>Risk Score: {risk_score}/100</span>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"<span style='font-size:0.85em;color:#555;'>{seg_label}</span>",
        unsafe_allow_html=True,
    )


def render_confidence(result: dict):
    conf  = result.get("analysis_confidence", "unknown")
    note  = result.get("confidence_note", "")
    msg   = f"**Analysis confidence: {conf.upper()}**" + (f"  \n{note}" if note else "")
    if conf == "high":
        st.success(msg)
    elif conf in ("medium", "low"):
        st.warning(msg)
    else:
        st.info(msg)


def render_models_status(result: dict):
    used    = result.get("models_used", [])
    unavail = result.get("models_unavailable", [])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Models used**")
        for m in used:
            st.markdown(f"  🟢 `{m}`")
    with col_b:
        st.markdown("**Models unavailable**")
        if not unavail:
            st.markdown("  *(none — all models responded)*")
        for m in unavail:
            st.markdown(f"  🔴 `{m['model']}` — {m.get('reason','')[:80]}")


def render_risk_score(result: dict):
    raw = result.get("raw_ml_predictions", {})
    rs  = raw.get("risk_score_analysis", {})
    if not isinstance(rs, dict):
        st.info(f"Risk Score Analysis: {rs}")
        return

    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Risk Score", f"{rs.get('score', '?')} / 100")
        st.markdown(sev_badge(rs.get("risk_level", "?")), unsafe_allow_html=True)
    with c2:
        factors = rs.get("factors_present", [])
        if factors:
            st.markdown("**Active risk factors:**")
            for f in factors:
                st.markdown(f"  • `{f}`")
        else:
            st.markdown("*No risk factors active.*")


def render_care_gaps(result: dict):
    raw = result.get("raw_ml_predictions", {})
    cg  = raw.get("care_gap_analysis", {})
    if not isinstance(cg, dict):
        st.info(f"Care Gap Analysis: {cg}")
        return

    gaps = cg.get("gaps_found", [])
    met  = cg.get("gaps_met", [])

    total   = cg.get("total_measures_checked", "?")
    n_gaps  = cg.get("gap_count", len(gaps))
    highest = cg.get("highest_severity", "none")

    st.markdown(
        f"**{n_gaps} gap(s)** found out of {total} measures  ·  "
        f"Highest severity: {sev_badge(highest)}",
        unsafe_allow_html=True,
    )

    if gaps:
        sev_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "none": 4}
        sorted_gaps = sorted(gaps, key=lambda g: sev_order.get(g.get("severity", "none"), 9))

        df = pd.DataFrame([
            {
                "ID":          g["id"],
                "Name":        g["name"],
                "Severity":    g["severity"].upper(),
                "Description": g.get("description", ""),
            }
            for g in sorted_gaps
        ])

        sev_colours = {
            "CRITICAL": "background-color:#f8d7da;color:#721c24;font-weight:600",
            "HIGH":     "background-color:#fdebd0;color:#7d4b00;font-weight:600",
            "MODERATE": "background-color:#fef9e7;color:#7d6608",
            "LOW":      "background-color:#d5f5e3;color:#1e5631",
        }

        def style_sev(val):
            return sev_colours.get(val, "")

        styled = df.style.applymap(style_sev, subset=["Severity"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.success("All measures met — no care gaps identified.")

    if met:
        with st.expander(f"✅ {len(met)} measures met"):
            st.markdown(", ".join(f"`{m['id']}`" for m in met))


def render_segmentation(result: dict):
    raw = result.get("raw_ml_predictions", {})
    seg = raw.get("patient_segmentation", {})
    if not isinstance(seg, dict):
        st.info(f"Segmentation: {seg}")
        return

    st.markdown(f"**Segment:** `{seg.get('segment_id', '?')}` — **{seg.get('segment_label', '?')}**")
    st.caption(seg.get("description", ""))
    features = seg.get("features_used", {})
    if features:
        cols = st.columns(4)
        items = list(features.items())
        for i, (k, v) in enumerate(items):
            cols[i % 4].metric(k.replace("_", " ").title(), str(v))

def render_agent_reasoning(result: dict):
    ar = result.get("agent_reasoning", {})
    if not isinstance(ar, dict):
        st.info("Agent reasoning not available.")
        return
    if "error" in ar:
        st.error(f"Ollama reasoning failed: {ar['error']}")
        if "note" in ar:
            st.caption(ar["note"])
        return

    fields = [
        ("quality_gaps_present",      "Quality Gaps Present"),
        ("most_significant_gap",      "Most Significant Gap"),
        ("risk_level",                "Risk Level Assessment"),
        ("patient_segment",           "Patient Segment"),
        ("model_consensus_summary",   "Model Consensus Summary"),
        ("cross_model_relationships", "Cross-Model Relationships"),
        ("priority_gap",              "Priority Gap"),
        ("supporting_evidence",       "Supporting Evidence"),
    ]
    for key, label in fields:
        val = ar.get(key)
        if val:
            st.markdown(f"**{label}**")
            st.markdown(f"> {val}")
            st.markdown("")


def render_recommendations(result: dict):
    recs = result.get("recommendations", [])
    if not recs:
        st.info("No recommendations returned.")
        return

    for i, rec in enumerate(recs, 1):
        action    = rec.get("action", "")
        rationale = rec.get("rationale", "")
        urgency   = rec.get("urgency", "routine").lower()
        with st.expander(
            f"{urgency_badge(urgency)} &nbsp; {action[:80]}{'…' if len(action) > 80 else ''}",
            expanded=(urgency == "immediate"),
        ):
            st.markdown(f"**Action:** {action}")
            st.markdown(f"**Rationale:** {rationale}")
            st.markdown(
                f"**Urgency:** {urgency_badge(urgency)}",
                unsafe_allow_html=True,
            )


def render_result(result: dict):
    render_header(result)
    st.markdown("---")

    render_confidence(result)

    st.markdown("#### 🔌 Model Status")
    render_models_status(result)
    st.markdown("---")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("#### 🎯 Risk Score")
        render_risk_score(result)
        st.markdown("---")
        st.markdown("#### 👤 Patient Segment")
        render_segmentation(result)

    with col_right:
        st.markdown("#### 🩺 Care Gaps")
        render_care_gaps(result)

    st.markdown("---")
    st.markdown("#### 🧠 Agent Reasoning")
    render_agent_reasoning(result)

    st.markdown("---")
    st.markdown("#### 💊 Recommendations")
    render_recommendations(result)

    st.markdown("---")
    with st.expander("🗂️ View raw JSON response"):
        st.json(result)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("🏥 Quality Gap Analysis Agent")
    st.caption(
        "Form → API → Ollama reasoning → structured report. "
        "Backend must be running on **localhost:8000** before submitting."
    )

    # Health check row
    hcol1, hcol2 = st.columns([6, 1])
    with hcol2:
        refresh = st.button("↻ Refresh status")

    # Only re-check health when page loads or refresh is clicked
    if "health_checked" not in st.session_state or refresh:
        st.session_state["health_checked"] = True

    with hcol1:
        backend_ok = render_health_banner()

    if not backend_ok:
        st.stop()

    # Sidebar form
    patient_id, payload = render_sidebar()

    # Result area — show last result if available, clear on new submit
    if patient_id is not None and payload is not None:
        # New submission
        with st.spinner("Running analysis — this can take up to a minute while Ollama reasons…"):
            status_code, body = run_analysis(patient_id, payload)

        if status_code == 0:
            if body == "connection_refused":
                st.error(
                    "❌ **Connection refused.** The backend is not running.\n\n"
                    "Start it with:\n```\npython -m uvicorn main:app --host 0.0.0.0 --port 8000\n```"
                )
            elif body == "timeout":
                st.error(
                    "⏱️ **Request timed out** after 180 s. "
                    "Ollama may still be processing. "
                    "Check the backend terminal for progress, then retry."
                )
            else:
                st.error(f"❌ Unexpected error: {body}")
        elif status_code != 200:
            st.error(f"❌ API returned HTTP {status_code}")
            st.json(body)
        else:
            st.session_state["last_result"] = body
            st.session_state["last_patient_id"] = patient_id

    # Render the most recent successful result
    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"])
    else:
        st.info(
            "👈 Fill in the patient form in the sidebar and click **Run Analysis** to begin."
        )


if __name__ == "__main__":
    main()
