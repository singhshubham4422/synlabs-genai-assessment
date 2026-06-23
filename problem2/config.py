import os
from dotenv import load_dotenv

# Load env variables from a .env file if present
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-2.5-flash")
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
LOG_DIR = os.getenv("LOG_DIR", "./logs")
REPORT_DIR = os.getenv("REPORT_DIR", "./reports")
