"""
Check Ollama service reachability and available models.
"""
import urllib.request
import json
import socket

def tcp_check(host, port, timeout=3):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False

print("=== Ollama Reachability ===")
reachable = tcp_check("localhost", 11434)
print(f"TCP connect localhost:11434 -> {'OK' if reachable else 'FAILED'}")

if not reachable:
    print("\nOllama is NOT running.")
    print("Start it with:  ollama serve")
    print("Then pull a model with:  ollama pull llama3")
    exit(1)

# Try /api/tags to list models
print("\n=== Available Models ===")
try:
    req = urllib.request.Request(
        "http://localhost:11434/api/tags",
        headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
        models = data.get("models", [])
        if not models:
            print("No models pulled yet.")
            print("Pull one with:  ollama pull llama3")
        else:
            for m in models:
                size_gb = m.get("size", 0) / 1e9
                print(f"  {m['name']:40s}  {size_gb:.1f} GB")
except Exception as e:
    print(f"ERROR querying /api/tags: {e}")

# Try /api/version
print("\n=== Ollama Version ===")
try:
    req = urllib.request.Request("http://localhost:11434/api/version")
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read().decode())
        print(f"Version: {data.get('version', 'unknown')}")
except Exception as e:
    print(f"ERROR: {e}")
