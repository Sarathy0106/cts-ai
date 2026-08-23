import os
from dotenv import load_dotenv

load_dotenv()

ML_API_BASE_URL = os.getenv("ML_API_BASE_URL", "http://54.237.50.176:8000").rstrip("/")
SEGMENTATION_API_URL = os.getenv("SEGMENTATION_API_URL", "http://54.157.160.227:8000").rstrip("/")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

BACKEND_API_BASE_URL = os.getenv("BACKEND_API_BASE_URL", "http://localhost:8000")
