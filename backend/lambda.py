import logging
import json
from mangum import Mangum
from app.main import app

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 Lambda 处理程序，使用兼容的配置
handler = Mangum(
    app, 
    lifespan="off",
    api_gateway_base_path="/")


# 添加调试日志 - Lambda启动时将记录此信息
logger.info("Lambda处理程序已初始化") 