# 基础镜像
FROM python:3.11-slim

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc \
    rustc \
    cargo \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 强制升级 pip（避免卸载报错）
RUN python3 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade --ignore-installed pip

# 设置工作目录
WORKDIR /var/task

# 复制代码和 requirements
COPY . .

# 安装依赖
RUN pip install primp==0.15.0 duckduckgo-search==8.0.1 && \
    pip install -r requirements.txt

# 设置环境变量（告诉 Lambda 用哪个模块和函数作为 handler）
ENV _HANDLER=lambda.handler

# 启动 runtime interface client
CMD ["python3", "-m", "awslambdaric", "lambda.handler"]