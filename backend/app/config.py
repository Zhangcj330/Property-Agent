from dotenv import load_dotenv
import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    DOMAIN_API_KEY: str = os.getenv("REAL_ESTATE_API_KEY")
    
    # Add other configuration settings here
    BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

settings = Settings() 