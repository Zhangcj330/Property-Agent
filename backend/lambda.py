import logging
import json
from mangum import Mangum
from app.main import app
from pythonjsonlogger import jsonlogger  # Add this dependency to requirements.txt

# Configure logging - central configuration point
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Clear any existing handlers
for h in root_logger.handlers:
    root_logger.removeHandler(h)
root_logger.addHandler(handler)

# Get module-specific logger
logger = logging.getLogger(__name__)

# Create Lambda handler with compatible configuration
handler = Mangum(
    app, 
    lifespan="off",
    api_gateway_base_path="/")

# Add initialization log
logger.info("Lambda handler initialized") 