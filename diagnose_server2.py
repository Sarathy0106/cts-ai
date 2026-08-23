"""
Part 2: identify which file has the encoding issue, scan remaining files,
and attempt a uvicorn startup to capture its real output.
"""
import socket, sys, os, traceback, subprocess, time

def sep(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")

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
    ("asyncio.run(",        "synchronous event loop block"),
    ("socket.connect(",     "synchronous socket call at module level"),
    ("urllib.request.open", "synchronous HTTP call at module level"),
    ("requests.get(",       "synchronous HTTP call at module level"),
    ("requests.post(",      "synchronous HTTP call at module level"),
    (".connect_ex(",        "synchronous socket connect at module level"),
    ("@app.on_event(",      "FastAPI startup/shutdown event handler"),
    ("lifespan",            "FastAPI lifespan handler"),
    ("startup",             "potential startup hook"),
]

# ── 1. Identify which files have encoding issues ──────────────────────────────
sep("1. File encoding scan")
bad_files = []
for fpath in files_to_scan:
    if not os.path.exists(fpath):
        print(f"  {fpath}: NOT FOUND")
        continue
    try:
        with open(fpath, encoding="utf-8") as f:
            f.read()
        print(f"  {fpath}: OK (utf-8)")
    except UnicodeDecodeError as e:
        bad_files.append(fpath)
        print(f"  {fpath}: *** ENCODING ERROR — {e} ***")
        # Show hex around the bad byte
        with open(fpath, "rb") as fb:
            raw = fb.read()
        pos = e.start
        print(f"    Bad byte at position {pos}: 0x{raw[pos]:02X}")
        print(f"    Context (bytes {max(0,pos-30)}..{pos+30}): {raw[max(0,pos-30):pos+30]}")

# ── 2. Scan clean files for blocking patterns ─────────────────────────────────
sep("2. Blocking-code scan (utf-8 readable files only)")
for fpath in files_to_scan:
    if fpath in bad_files or not os.path.exists(fpath):
        continue
    try:
        with open(fpath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  {fpath}: skip ({e})")
        continue
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
            print(f"    line {lineno:4d}: [{label}]  {content}")
    else:
        print(f"  {fpath}: clean")

# ── 3. Attempt uvicorn startup, capture output for 8 seconds ─────────────────
sep("3. uvicorn startup test (8 second capture)")
print("  Launching: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
print("  Waiting 8 seconds for output...\n")

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app",
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=os.getcwd(),
    text=True,
    encoding="utf-8",
    errors="replace",
)

# Read output for up to 8 seconds
start = time.time()
output_lines = []
while time.time() - start < 8:
    line = proc.stdout.readline()
    if line:
        output_lines.append(line.rstrip())
        print(f"  [uvicorn] {line.rstrip()}")
    elif proc.poll() is not None:
        # process exited
        remaining = proc.stdout.read()
        if remaining:
            for l in remaining.splitlines():
                output_lines.append(l)
                print(f"  [uvicorn] {l}")
        break

exit_code = proc.poll()
print(f"\n  Process status after 8s: {'still running' if exit_code is None else f'EXITED with code {exit_code}'}")

# ── 4. Check port 8000 after startup attempt ──────────────────────────────────
sep("4. Port 8000 status after startup attempt")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
result = s.connect_ex(("127.0.0.1", 8000))
s.close()
print(f"  port 8000: {'LISTENING' if result == 0 else f'not listening (errno {result})'}")

# Kill the test uvicorn process if still running
if proc.poll() is None:
    proc.terminate()
    print("  (test uvicorn process terminated)")

print("\nDiagnostics complete.")
