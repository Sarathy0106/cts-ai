"""Syntax + import check for all edited files."""
import ast, sys, importlib, traceback

FILES = [
    "agent/health_check.py",
    "agent/orchestrator.py",
    "agent/context_builder.py",
    "agent/reasoning_prompt.py",
    "tools/ganesh_tool.py",
    "tools/prediction_tool.py",
    "tools/star_impact_tool.py",
    "main.py",
]

print("=== Syntax check ===")
all_ok = True
for f in FILES:
    try:
        with open(f, encoding="utf-8") as fh:
            src = fh.read()
        ast.parse(src)
        print(f"  OK  {f}")
    except SyntaxError as e:
        print(f"  FAIL {f}: SyntaxError line {e.lineno}: {e.msg}")
        all_ok = False
    except Exception as e:
        print(f"  FAIL {f}: {e}")
        all_ok = False

print()
print("=== Import check ===")
# Flush cached pyc so we test the actual source
for key in list(sys.modules.keys()):
    if any(key.startswith(p) for p in ("agent", "tools", "models", "main")):
        del sys.modules[key]

IMPORTS = [
    ("agent.health_check",    ["check_remote_services", "ServiceStatus", "is_reachable"]),
    ("agent.context_builder", ["build_context"]),
    ("agent.reasoning_prompt",["build_messages"]),
    ("agent.ollama_client",   ["chat", "parse_json_response", "OllamaError"]),
    ("tools.ganesh_tool",     ["risk_prediction_model_tool", "call_risk_prediction_model"]),
    ("tools.prediction_tool", ["prioritization_model_tool", "call_prioritization_model"]),
    ("tools.star_impact_tool",["star_impact_model_tool", "call_star_impact_model"]),
    ("agent.orchestrator",    ["run_analysis", "TOOL_REGISTRY"]),
    ("main",                  ["app"]),
]

for mod_name, attrs in IMPORTS:
    try:
        mod = importlib.import_module(mod_name)
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            print(f"  FAIL {mod_name}: missing attrs {missing}")
            all_ok = False
        else:
            print(f"  OK  {mod_name}: {attrs}")
    except Exception:
        print(f"  FAIL {mod_name}:")
        traceback.print_exc()
        all_ok = False

print()
print("=== Tool registry check ===")
try:
    from agent.orchestrator import TOOL_REGISTRY
    expected = {
        "risk_score_analysis", "care_gap_analysis", "patient_segmentation",
        "risk_prediction_model", "prioritization_model", "star_impact_model",
    }
    actual = set(TOOL_REGISTRY.keys())
    extra   = actual - expected
    missing = expected - actual
    if missing:
        print(f"  FAIL: missing keys: {missing}")
        all_ok = False
    if extra:
        print(f"  WARN: unexpected extra keys: {extra}")
    if not missing:
        print(f"  OK  registry keys: {sorted(actual)}")
except Exception:
    traceback.print_exc()
    all_ok = False

print()
print("=== ServiceStatus field check ===")
try:
    from agent.health_check import ServiceStatus
    s = ServiceStatus(name="test", url="https://x.com", reachable=True, detail="HTTP 200")
    assert not hasattr(s, "host"), "ServiceStatus still has old 'host' field"
    assert not hasattr(s, "port"), "ServiceStatus still has old 'port' field"
    assert hasattr(s, "url"),    "ServiceStatus missing 'url'"
    assert hasattr(s, "detail"), "ServiceStatus missing 'detail'"
    print(f"  OK  ServiceStatus fields: {s._fields}")
except Exception:
    traceback.print_exc()
    all_ok = False

print()
print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
