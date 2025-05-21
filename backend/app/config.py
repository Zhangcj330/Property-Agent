import os
import json
import logging
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Load .env file if exists (useful for local development)
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.debug(f"Loaded environment variables from {env_path}")
else:
    logger.debug(".env file not found, skipping local environment loading")

# Firebase config loader
def load_firebase_config() -> Dict:
    """
    Load Firebase credentials from environment variable or fallback file.
    Returns:
        dict: Parsed service account credentials
    """
    credentials_str = os.getenv("FIREBASE_CREDENTIALS") or os.getenv("FIREBASE_CREDENTIALS_JSON")

    if credentials_str:
        try:
            credentials = json.loads(credentials_str)
            logger.info("Firebase credentials successfully loaded from environment variable.")
            return credentials
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Firebase credentials from environment variable: {e}")

    # Fallback to file
    fallback_path = BASE_DIR / "Firebase_key.json"
    if fallback_path.exists():
        try:
            with open(fallback_path, "r") as f:
                credentials = json.load(f)
                logger.info(f"Firebase credentials successfully loaded from file: {fallback_path}")
                return credentials
        except Exception as e:
            logger.warning(f"Failed to load Firebase credentials from file: {e}")
    else:
        logger.warning(f"Firebase_key.json not found at: {fallback_path}")

    return {}

# Load Firebase config once at module level
FIREBASE_CONFIG = load_firebase_config()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DOMAIN_API_KEY: str = os.getenv("DOMAIN_API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    FIREBASE_CONFIG: dict = FIREBASE_CONFIG
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")

settings = Settings() 