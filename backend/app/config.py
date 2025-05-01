from dotenv import load_dotenv
import os
import json
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

# Firebase配置 - 优先从环境变量读取，如果不存在则尝试从文件读取
FIREBASE_CONFIG = {}
try:
    # 首先尝试从环境变量读取JSON字符串
    firebase_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if firebase_creds_json:
        FIREBASE_CONFIG = json.loads(firebase_creds_json)
        print("从环境变量加载Firebase凭证")
    else:
        # 如果环境变量不存在，尝试从文件读取
        FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", BASE_DIR / 'Firebase_key.json')
        with open(FIREBASE_KEY_PATH) as f:
            FIREBASE_CONFIG = json.load(f)
            print(f"从文件加载Firebase凭证: {FIREBASE_KEY_PATH}")
except Exception as e:
    print(f"警告: 无法加载Firebase凭证: {str(e)}")
    # 在开发环境中，如果无法加载凭证可能需要终止
    # 在生产环境中，可能需要继续但禁用Firebase相关功能
    # 根据你的需求调整这里的处理方式

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DOMAIN_API_KEY: str = os.getenv("DOMAIN_API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    FIREBASE_CONFIG: dict = FIREBASE_CONFIG

settings = Settings() 