import os
from dotenv import load_dotenv

load_dotenv()

# ML API Settings
ML_API_BASE_URL = os.getenv("ML_API_BASE_URL", "http://54.237.50.176:8000").rstrip("/")

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Internal backend API
BACKEND_API_BASE_URL = os.getenv("BACKEND_API_BASE_URL", "http://localhost:8000")
