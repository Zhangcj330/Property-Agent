from dotenv import load_dotenv
import os
import json
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

# Load Firebase credentials from json file
FIREBASE_KEY_PATH = BASE_DIR / 'Firebase_key.json'
with open(FIREBASE_KEY_PATH) as f:
    FIREBASE_CONFIG = json.load(f)

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    DOMAIN_API_KEY: str = os.getenv("DOMAIN_API_KEY")
    BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    FIREBASE_CONFIG: dict = FIREBASE_CONFIG

settings = Settings() 