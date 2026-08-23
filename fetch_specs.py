"""
Schema inspection for remote ML services.
Tries /openapi.json and /docs for each service URL.
Edit the SERVICES list to point at new endpoints; run to inspect schemas.
"""
import urllib.request
import urllib.error
import json
import ssl

# Services to inspect — edit this list when endpoints change
SERVICES = [
    {
        "name": "Risk Prediction Model",
        "base": "https://api.sidanex.com/cts-ml1/",
    },
    {
        "name": "Prioritization Model",
        "base": "https://api.sidanex.com/cts-ml3/",
    },
    {
        "name": "Star Impact Model",
        "base": "https://api.sidanex.com/cts-ml4/",
    },
]

TIMEOUT = 15


def fetch_url(url: str) -> tuple[int | None, str]:
    """Returns (http_status, body_or_error_string)."""
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json, text/html"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = "(could not read error body)"
        return e.code, body
    except urllib.error.URLError as e:
        return None, f"URLError: {e.reason}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def inspect_service(name: str, base: str) -> dict:
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  Base URL: {base}")
    print("="*65)

    result = {"name": name, "base": base, "openapi": None, "docs": None}

    # Try /openapi.json
    for path in ["openapi.json", "docs"]:
        url = base.rstrip("/") + "/" + path
        print(f"\n  Trying: {url}")
        status, body = fetch_url(url)
        print(f"  HTTP status: {status}")

        if status is None:
            print(f"  Result: UNREACHABLE — {body}")
            result[path.replace(".json", "").replace("/", "")] = {
                "reachable": False, "error": body
            }
            continue

        if status == 401:
            print("  Result: 401 UNAUTHORIZED — authentication required")
            result[path.replace(".json", "").replace("/", "")] = {
                "reachable": True, "status": 401, "auth_required": True
            }
            continue

        if status == 403:
            print("  Result: 403 FORBIDDEN — access denied")
            result[path.replace(".json", "").replace("/", "")] = {
                "reachable": True, "status": 403, "auth_required": True
            }
            continue

        if status == 404:
            print("  Result: 404 NOT FOUND")
            result[path.replace(".json", "").replace("/", "")] = {
                "reachable": True, "status": 404
            }
            continue

        if status == 200:
            # Try to parse as JSON
            try:
                data = json.loads(body)
                print(f"  Result: 200 OK — valid JSON ({len(body)} chars)")
                print(json.dumps(data, indent=2)[:3000])
                result[path.replace(".json", "").replace("/", "")] = {
                    "reachable": True, "status": 200, "data": data
                }
                # Save raw spec to file
                fname = name.lower().replace(" ", "_") + "_" + path.replace("/", "_")
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved to: {fname}")
            except json.JSONDecodeError:
                print(f"  Result: 200 OK — not JSON (HTML/text, {len(body)} chars)")
                print(f"  First 500 chars: {body[:500]}")
                result[path.replace(".json", "").replace("/", "")] = {
                    "reachable": True, "status": 200, "content_type": "html/text",
                    "preview": body[:500]
                }
        else:
            print(f"  Result: HTTP {status} — {body[:300]}")
            result[path.replace(".json", "").replace("/", "")] = {
                "reachable": True, "status": status, "body_preview": body[:300]
            }

    return result


if __name__ == "__main__":
    print("Schema inspection for remote ML services")
    print(f"Timeout per request: {TIMEOUT}s\n")

    all_results = []
    for svc in SERVICES:
        r = inspect_service(svc["name"], svc["base"])
        all_results.append(r)

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print("="*65)
    for r in all_results:
        openapi = r.get("openapi", {}) or {}
        docs    = r.get("docs", {}) or {}
        reachable = openapi.get("reachable") or docs.get("reachable")
        auth      = openapi.get("auth_required") or docs.get("auth_required")
        status_str = "AUTH REQUIRED" if auth else ("REACHABLE" if reachable else "UNREACHABLE")
        print(f"  {r['name']:40s} {status_str}")
