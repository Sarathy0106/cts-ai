"""
Full server diagnostics — run before touching any code.
Covers: port binding, import-time blocking, startup events, module-level code.
"""
import socket
import sys
import importlib
import traceback
import subprocess
import os

def sep(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")

# ── 1. Port occupancy ─────────────────────────────────────────────────────────
sep("1. Port occupancy check (8000, 8001, 11434)")
for port in [8000, 8001, 11434]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    result = s.connect_ex(("127.0.0.1", port))
    s.close()
    status = "IN USE / listening" if result == 0 else f"not listening (errno {result})"
    print(f"  port {port}: {status}")

# ── 2. Try to get process on port 8000 via psutil if available ────────────────
sep("2. Process holding port 8000 (psutil)")
try:
    import psutil
    found = False
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == 8000:
            pid = conn.pid
            try:
                proc = psutil.Process(pid)
                print(f"  PID {pid}: {proc.name()} — {' '.join(proc.cmdline())}")
            except Exception:
                print(f"  PID {pid}: (could not read process info)")
            found = True
    if not found:
        print("  Nothing bound to port 8000.")
except ImportError:
    print("  psutil not installed — skipping (install with: pip install psutil)")

# ── 3. Dry-run import of main.py — catch any import-time errors ───────────────
sep("3. Import-time analysis of main.py")
print("  Importing main module (without running anything)...")
try:
    # Remove cached version if present
    for key in list(sys.modules.keys()):
        if key.startswith(("main", "agent", "tools", "models")):
            del sys.modules[key]

    import main as app_module
    print("  [OK] main.py imported without errors.")
    print(f"  Routes registered: {[r.path for r in app_module.app.routes]}")
except Exception:
    print("  [FAIL] main.py import raised an exception:")
    traceback.print_exc()

# ── 4. Scan for module-level blocking code ────────────────────────────────────
sep("4. Module-level / startup-event code scan")

files_to_scan = [
    "main.py",
    "agent/orchestrator.py",
    "agent/health_check.py",
    "agent/ollama_client.py",
    "agent/context_builder.py",
    "agent/reasoning_prompt.py",
    "tools/ganesh_tool.py",
    "tools/prediction_tool.py",
    "tools/risk_score_tool.py",
    "tools/care_gap_tool.py",
    "tools/segmentation_tool.py",
]

blocking_patterns = [
    ("asyncio.run(",         "synchronous event loop block"),
    ("socket.connect(",      "synchronous socket call at module level"),
    ("urllib.request.open",  "synchronous HTTP call at module level"),
    ("requests.get(",        "synchronous HTTP call at module level"),
    ("requests.post(",       "synchronous HTTP call at module level"),
    (".connect_ex(",         "synchronous socket connect at module level"),
    ("@app.on_event(",       "FastAPI startup/shutdown event handler"),
    ("@app.router.on_event", "FastAPI startup/shutdown event handler"),
    ("lifespan",             "FastAPI lifespan handler"),
    ("startup",              "potential startup hook"),
]

for fpath in files_to_scan:
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        lines = f.readlines()
    hits = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, label in blocking_patterns:
            if pattern in line:
                hits.append((i, stripped[:100], label))
    if hits:
        print(f"\n  {fpath}:")
        for lineno, content, label in hits:
            print(f"    line {lineno:4d}: [{label}]")
            print(f"             {content}")
    else:
        print(f"  {fpath}: clean")

# ── 5. Quick uvicorn dry-run — just import check ─────────────────────────────
sep("5. uvicorn importable + version")
try:
    import uvicorn
    print(f"  uvicorn version: {uvicorn.__version__}")
    print("  [OK] uvicorn importable.")
except ImportError as e:
    print(f"  [FAIL] uvicorn import error: {e}")

# ── 6. Check if 'ollama' SDK still imported in orchestrator ──────────────────
sep("6. Stale 'import ollama' check in orchestrator.py")
with open("agent/orchestrator.py") as f:
    orch_src = f.read()
if "import ollama" in orch_src and "ollama_client" not in orch_src:
    print("  [WARNING] orchestrator.py has bare 'import ollama' but no ollama_client reference.")
    print("  This could fail if the 'ollama' SDK package is not installed or conflicts.")
elif "import ollama" in orch_src:
    print("  [NOTE] orchestrator.py imports both 'ollama' SDK and 'ollama_client'.")
    print("  Lines containing 'import ollama':")
    for i, line in enumerate(orch_src.splitlines(), 1):
        if "import ollama" in line:
            print(f"    line {i}: {line.strip()}")
else:
    print("  [OK] No bare 'import ollama' — using ollama_client only.")

print("\nDiagnostics complete.")
