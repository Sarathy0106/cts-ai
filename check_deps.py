import importlib.util, sys
print("Python:", sys.version)
mods = ['fastapi','uvicorn','httpx','ollama','pydantic','pandas','sklearn','numpy','aiohttp']
for m in mods:
    spec = importlib.util.find_spec(m)
    print(m, "OK" if spec else "MISSING")
