# 构建阶段 - 移除，不再使用多阶段构建
FROM public.ecr.aws/lambda/python:3.11

# 设置工作目录
WORKDIR ${LAMBDA_TASK_ROOT}

# 仅安装必要的系统依赖
RUN yum install -y tar gzip && yum clean all

# 复制requirements.txt并安装依赖
COPY requirements.txt .

# 安装预编译的lxml而不是从源码编译
RUN pip install --no-cache-dir --platform manylinux2014_aarch64 --target=${LAMBDA_TASK_ROOT} --implementation cp --python-version 3.11 --only-binary=:all: --upgrade lxml

# 安装其他依赖
RUN pip install --no-cache-dir --target=${LAMBDA_TASK_ROOT} -r requirements.txt

# 创建必要的目录结构
RUN mkdir -p /app/secrets /app/data /app/app/data

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}
ENV PYTHONUNBUFFERED=1

# 设置 Lambda 处理程序
CMD ["lambda.handler"] 