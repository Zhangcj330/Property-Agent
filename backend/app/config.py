from dotenv import load_dotenv
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
try:
    env_path = BASE_DIR / '.env'
    load_dotenv(env_path)
    logger.debug(f"Loaded environment variables from {env_path}")
except ImportError:
    logger.debug("python-dotenv not available, skipping .env loading")

# Try to load Firebase credentials from environment variable first
firebase_credentials = None
if os.environ.get('FIREBASE_CREDENTIALS'):
    try:
        logger.info("Loading Firebase credentials from environment variable")
        firebase_credentials = json.loads(os.environ.get('FIREBASE_CREDENTIALS', '{}'))
    except json.JSONDecodeError:
        logger.warning("Failed to parse Firebase credentials from environment variable")

# If not loaded from environment variable, try to load from file
if not firebase_credentials:
    # Check if running in Docker (where the path would be /app/Firebase_key.json)
    docker_path = '/app/Firebase_key.json'
    local_path = str(BASE_DIR / 'Firebase_key.json')
    
    # Use environment variable if set, otherwise try Docker path first, then local path
    firebase_key_path = os.environ.get('FIREBASE_KEY_PATH')
    
    if not firebase_key_path:
        # Try Docker path first if it exists
        if os.path.exists(docker_path):
            firebase_key_path = docker_path
        else:
            firebase_key_path = local_path
    
    try:
        logger.info(f"Loading Firebase credentials from file: {firebase_key_path}")
        with open(firebase_key_path, 'r') as f:
            firebase_credentials = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load Firebase credentials from file: {str(e)}")

# Firebase configuration - First try loading from environment variables, then from file
FIREBASE_CONFIG = {}
try:
    # First try loading JSON string from environment variable
    firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if firebase_creds_json:
        FIREBASE_CONFIG = json.loads(firebase_creds_json)
        logger.info("Loaded Firebase credentials from environment variable")
    else:
        # If environment variable doesn't exist, try loading from file
        FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", BASE_DIR / 'Firebase_key.json')
        with open(FIREBASE_KEY_PATH) as f:
            FIREBASE_CONFIG = json.load(f)
            logger.info(f"Loaded Firebase credentials from file: {FIREBASE_KEY_PATH}")
except Exception as e:
    logger.warning(f"Warning: Failed to load Firebase credentials: {str(e)}")
    # In development, you might need to terminate if credentials can't be loaded
    # In production, you might continue but disable Firebase-related features
    # Adjust this handling based on your requirements

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DOMAIN_API_KEY: str = os.getenv("DOMAIN_API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    FIREBASE_CONFIG: dict = FIREBASE_CONFIG

settings = Settings() 