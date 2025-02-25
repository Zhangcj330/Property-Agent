from dotenv import load_dotenv
import os
from pathlib import Path

# Load test environment variables
load_dotenv(Path(__file__).parent / 'test.env')

TEST_GEMINI_API_KEY = os.getenv("TEST_GEMINI_API_KEY") 