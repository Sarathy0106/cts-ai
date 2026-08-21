import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ML API Settings
ML_API_BASE_URL = os.getenv("ML_API_BASE_URL", "http://54.237.50.176:8000").rstrip("/")
ML_PREDICT_SINGLE_URL = f"{ML_API_BASE_URL}/predict/single"
ML_PREDICT_BATCH_URL = f"{ML_API_BASE_URL}/predict/batch"

# Ollama Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Groq Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
