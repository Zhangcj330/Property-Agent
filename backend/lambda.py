import logging
from mangum import Mangum
import time
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
_mangum_handler = None

def get_app():
    global _app
    if _app is None:
        logger.info("Starting FastAPI app initialization")
        start_time = time.time()
        # Defer import of app.main to reduce cold start time
        from app.main import app
        _app = app
        elapsed = time.time() - start_time
        logger.info(f"FastAPI application initialized on first request in {elapsed:.2f}s")
    return _app

def get_lambda_handler():
    global _mangum_handler
    if _mangum_handler is None:
        logger.info("Creating Mangum handler")
        # Do not initialize the app here, just pass the factory function
        # This ensures the app is only initialized when the first request comes in
        _mangum_handler = Mangum(
            get_app(),
            lifespan="off",
            api_gateway_base_path="/"
        )
    return _mangum_handler

def handler(event, context):
    logger.info(f"Lambda event: {event}")
    mangum_handler = get_lambda_handler()
    response = mangum_handler(event, context)
    logger.info(f"Lambda response: {response}")
    return response