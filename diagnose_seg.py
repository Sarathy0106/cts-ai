"""
Diagnose segmentation non-determinism.
Checks: feature extraction, lambda evaluation, _SEGMENTS list identity between calls,
and whether the module-level PATIENT dict or _SEGMENTS list gets mutated.
"""
import sys, copy
from models.segmentation import _SEGMENTS, _extract_features, segment_patient

PATIENT = {
    "age": 68, "sex": "female",
    "conditions": ["diabetes", "hypertension"],
    "medications_count": 6,
    "ed_visits_90d": 1,
    "hospitalizations_180d": 0,
    "has_pcp": True,
    "med_adherence_pct": 55.0,
    "smoker": False,
    "bmi": 31.2,
}

print("=== Feature extraction with risk_score=55, care_gap_count=8 ===")
features = _extract_features(PATIENT, risk_score=55, care_gap_count=8)
for k, v in features.items():
    print(f"  {k}: {v}")

print()
print("=== Lambda evaluation for each segment ===")
for seg in _SEGMENTS:
    try:
        result = seg["criteria"](features)
    except Exception as e:
        result = f"ERROR: {e}"
    print(f"  {seg['id']:25s} -> {result}")

print()
print("=== 10 consecutive calls — should always return same segment ===")
results = set()
for i in range(10):
    # use a deep copy each time to rule out mutation of input dict
    r = segment_patient(copy.deepcopy(PATIENT), risk_score=55, care_gap_count=8)
    results.add(r["segment_id"])
    print(f"  call {i+1:2d}: {r['segment_id']}")

print()
if len(results) == 1:
    print(f"DETERMINISTIC — always returned: {results.pop()}")
else:
    print(f"NON-DETERMINISTIC — got multiple results: {results}")

print()
print("=== Check _SEGMENTS list identity across calls ===")
id1 = id(_SEGMENTS)
from models.segmentation import _SEGMENTS as _SEG2
id2 = id(_SEG2)
print(f"  Same object both imports: {id1 == id2}")
print(f"  _SEGMENTS length: {len(_SEGMENTS)}")
print(f"  Segment order: {[s['id'] for s in _SEGMENTS]}")

print()
print("=== Check if PATIENT dict is mutated after calls ===")
before = copy.deepcopy(PATIENT)
segment_patient(PATIENT, risk_score=55, care_gap_count=8)
after = PATIENT
mutated_keys = [k for k in before if before[k] != after.get(k)]
if mutated_keys:
    print(f"  MUTATION DETECTED on keys: {mutated_keys}")
else:
    print("  No mutation of patient dict detected.")
