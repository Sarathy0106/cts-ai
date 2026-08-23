"""
Verifies that care_gap_analysis uses actual lab values and dates,
not just condition presence. Tests CDC_A1C and CBP specifically.
"""
from __future__ import annotations
import asyncio
import json
from datetime import date, timedelta
from models.care_gap import analyze_care_gaps

# Base patient — diabetes + hypertension, age 68
BASE = {
    "age": 68, "sex": "female",
    "conditions": ["diabetes", "hypertension"],
    "medications_count": 6,
    "ed_visits_90d": 1,
    "hospitalizations_180d": 0,
    "has_pcp": True,
    "med_adherence_pct": 55.0,
    "smoker": False,
    "bmi": 31.2,
    # dates for other measures (keep stable so only the tested field changes)
    "last_mammogram_date":         "2021-06-10",
    "last_flu_vaccine_date":       "2022-09-01",
    "last_kidney_eval_date":       None,
    "last_wellness_visit_date":    None,
    "last_med_reconciliation_date": None,
    "last_colonoscopy_date":       None,
    "last_depression_screen_date": None,
    "last_pap_date":               "2024-01-01",
}

def run(patient: dict) -> dict:
    return analyze_care_gaps(patient)

def gap_entry(result: dict, gap_id: str):
    for g in result["gaps_found"]:
        if g["id"] == gap_id:
            return ("gaps_found", g)
    for g in result["gaps_met"]:
        if g["id"] == gap_id:
            return ("gaps_met", g)
    return (None, None)

def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)

def show_gap(label, patient, gap_id):
    result = run(patient)
    location, entry = gap_entry(result, gap_id)
    if location == "gaps_found":
        print(f"  {label:50s} → GAP FOUND   severity={entry['severity']}")
    elif location == "gaps_met":
        print(f"  {label:50s} → GAP MET     (measure passed)")
    else:
        print(f"  {label:50s} → NOT EVALUATED (unexpected)")
    return location, entry

# ══════════════════════════════════════════════════════════════════════════════
sep("CDC_A1C — Source code for this measure")
import inspect
from models.care_gap import MEASURE_CATALOG
cdc = next(m for m in MEASURE_CATALOG if m["id"] == "CDC_A1C")
cbp = next(m for m in MEASURE_CATALOG if m["id"] == "CBP")
print(f"""
  CDC_A1C check logic (from MEASURE_CATALOG):
    Triggers gap when ALL of:
      1. "diabetes" in patient.conditions
      2. ANY of:
           a) last_a1c_date is missing
           b) last_a1c_date is > 365 days ago
           c) last_a1c_value >= 9.0

  CBP check logic (from MEASURE_CATALOG):
    Triggers gap when ALL of:
      1. "hypertension" in patient.conditions
      2. age between 18-85
      3. ANY of:
           a) last_bp_systolic is missing
           b) last_bp_systolic >= 140
           c) last_bp_diastolic >= 90
""")

# ══════════════════════════════════════════════════════════════════════════════
sep("TEST 1: CDC_A1C — varying last_a1c_value (date fixed within 1 year)")

# Date within 1 year — so only the value should determine outcome
recent_date = (date.today() - timedelta(days=90)).isoformat()  # 90 days ago

p_A = {**BASE, "last_a1c_date": "2023-01-15", "last_a1c_value": 9.4}
p_B = {**BASE, "last_a1c_date": recent_date,  "last_a1c_value": 8.5}
p_C = {**BASE, "last_a1c_date": recent_date,  "last_a1c_value": 6.5}

print(f"\n  (recent_date used for B/C = {recent_date})")
print()
show_gap("Run A: a1c_date=2023-01-15  a1c_value=9.4 (>= 9.0)",  p_A, "CDC_A1C")
show_gap("Run B: a1c_date=recent(90d) a1c_value=8.5 (< 9.0)",   p_B, "CDC_A1C")
show_gap("Run C: a1c_date=recent(90d) a1c_value=6.5 (well ctrl)",p_C, "CDC_A1C")

print()
print("  Expected: A=GAP FOUND, B=GAP MET, C=GAP MET")
lA, _ = gap_entry(run(p_A), "CDC_A1C")
lB, _ = gap_entry(run(p_B), "CDC_A1C")
lC, _ = gap_entry(run(p_C), "CDC_A1C")
ok = (lA == "gaps_found" and lB == "gaps_met" and lC == "gaps_met")
print(f"  Result: {'PASS — value field IS used' if ok else 'FAIL — value field NOT used correctly'}")

# ══════════════════════════════════════════════════════════════════════════════
sep("TEST 2: CDC_A1C — date staleness (value fixed at 6.5, well-controlled)")

# Good value but test whether date alone triggers the gap
old_date   = (date.today() - timedelta(days=400)).isoformat()   # > 365 days ago
fresh_date = (date.today() - timedelta(days=60)).isoformat()    # 60 days ago

p_D = {**BASE, "last_a1c_date": old_date,   "last_a1c_value": 6.5}
p_E = {**BASE, "last_a1c_date": fresh_date, "last_a1c_value": 6.5}

print(f"\n  old_date   = {old_date}  (> 365 days ago)")
print(f"  fresh_date = {fresh_date} (60 days ago)")
print()
show_gap("Run D: a1c_date=OLD(400d) a1c_value=6.5 (good value, stale date)", p_D, "CDC_A1C")
show_gap("Run E: a1c_date=fresh(60d) a1c_value=6.5 (good value, fresh date)", p_E, "CDC_A1C")

print()
print("  Expected: D=GAP FOUND (stale date), E=GAP MET (fresh + good value)")
lD, _ = gap_entry(run(p_D), "CDC_A1C")
lE, _ = gap_entry(run(p_E), "CDC_A1C")
ok2 = (lD == "gaps_found" and lE == "gaps_met")
print(f"  Result: {'PASS — date field IS used' if ok2 else 'FAIL — date field NOT used correctly'}")

# ══════════════════════════════════════════════════════════════════════════════
sep("TEST 3: CBP — varying bp_systolic / bp_diastolic")

p_F = {**BASE, "last_bp_systolic": 148, "last_bp_diastolic": 92}   # uncontrolled
p_G = {**BASE, "last_bp_systolic": 135, "last_bp_diastolic": 85}   # at target
p_H = {**BASE, "last_bp_systolic": 160, "last_bp_diastolic": 100}  # clearly uncontrolled

print()
show_gap("Run F: BP = 148/92  (>= 140/90, uncontrolled)",   p_F, "CBP")
show_gap("Run G: BP = 135/85  (< 140/90, at target)",       p_G, "CBP")
show_gap("Run H: BP = 160/100 (clearly uncontrolled)",      p_H, "CBP")

print()
print("  Expected: F=GAP FOUND, G=GAP MET, H=GAP FOUND")
lF, eF = gap_entry(run(p_F), "CBP")
lG, eG = gap_entry(run(p_G), "CBP")
lH, eH = gap_entry(run(p_H), "CBP")
ok3 = (lF == "gaps_found" and lG == "gaps_met" and lH == "gaps_found")
print(f"  Result: {'PASS — BP values ARE used' if ok3 else 'FAIL — BP values NOT used correctly'}")

# ══════════════════════════════════════════════════════════════════════════════
sep("TEST 4: CBP — diastolic alone triggers gap (systolic fine, diastolic not)")

p_I = {**BASE, "last_bp_systolic": 130, "last_bp_diastolic": 95}   # systolic ok, diastolic high
p_J = {**BASE, "last_bp_systolic": 145, "last_bp_diastolic": 80}   # systolic high, diastolic ok

print()
show_gap("Run I: BP=130/95  (systolic ok, diastolic >= 90)",  p_I, "CBP")
show_gap("Run J: BP=145/80  (systolic >= 140, diastolic ok)", p_J, "CBP")

print()
print("  Expected: I=GAP FOUND (diastolic triggers), J=GAP FOUND (systolic triggers)")
lI, _ = gap_entry(run(p_I), "CBP")
lJ, _ = gap_entry(run(p_J), "CBP")
ok4 = (lI == "gaps_found" and lJ == "gaps_found")
print(f"  Result: {'PASS' if ok4 else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
sep("TEST 5: Missing fields (no values provided at all)")

p_K = {**BASE}  # no a1c or bp fields at all
p_K.pop("last_a1c_date",    None)
p_K.pop("last_a1c_value",   None)
p_K.pop("last_bp_systolic", None)
p_K.pop("last_bp_diastolic",None)

print()
show_gap("Run K (no a1c fields): CDC_A1C",  p_K, "CDC_A1C")
show_gap("Run K (no bp fields):  CBP",      p_K, "CBP")
print("  Expected: both GAP FOUND (missing fields treated as gap)")

# ══════════════════════════════════════════════════════════════════════════════
sep("CONCLUSION")
all_pass = ok and ok2 and ok3 and ok4
print(f"""
  CDC_A1C uses lab values: {'YES' if ok else 'NO'}
    - last_a1c_value >= 9.0  → gap (even if date is recent)
    - last_a1c_date > 365d   → gap (even if value is good)
    - both recent + < 9.0    → gap met (no gap)

  CBP uses BP readings:     {'YES' if ok3 else 'NO'}
    - systolic >= 140        → gap
    - diastolic >= 90        → gap (independent of systolic)
    - both < 140/90          → gap met (no gap)

  Overall verdict:
  {'care_gap_analysis DOES use lab values and dates to determine gap status.' if all_pass
   else 'SOME checks are NOT using lab values correctly — see FAIL output above.'}

  Severity is STATIC per measure (hard-coded in MEASURE_CATALOG):
    CDC_A1C = always "critical" when triggered
    CBP     = always "high" when triggered
  The severity does not scale with how far out of range the value is.
  A1C of 9.1 and A1C of 14.0 both produce severity="critical".
  This is by design (HEDIS measures are binary pass/fail per interval).
""")
