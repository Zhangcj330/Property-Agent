import logging
from mangum import Mangum
from app.main import app
from pythonjsonlogger import jsonlogger

# ---------- Configure Logging ----------
log_stream_handler = logging.StreamHandler()
log_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
log_stream_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for h in root_logger.handlers:
    root_logger.removeHandler(h)
root_logger.addHandler(log_stream_handler)

logger = logging.getLogger(__name__)

# ---------- App and Handler ----------
_app = None

def get_app():
    global _app
    if _app is None:
        _app = app
        logger.info("FastAPI application initialized on first request")
    return _app

def get_lambda_handler():
    return Mangum(
        get_app(),
        lifespan="off",
        api_gateway_base_path="/"
    )

def handler(event, context):
    mangum_handler = get_lambda_handler()
    logger.info("Processing Lambda request")
    return mangum_handler(event, context)