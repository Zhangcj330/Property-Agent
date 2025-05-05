# Build stage - Removed multi-stage build approach
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Install only necessary system dependencies
RUN yum install -y tar gzip && yum clean all

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# Install pre-compiled lxml instead of building from source
RUN pip install --no-cache-dir --platform manylinux2014_aarch64 --target=${LAMBDA_TASK_ROOT} --implementation cp --python-version 3.11 --only-binary=:all: --upgrade lxml

# Install other dependencies
RUN pip install --no-cache-dir --target=${LAMBDA_TASK_ROOT} -r requirements.txt

# Create necessary directory structure
RUN mkdir -p /app/secrets /app/data /app/app/data

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}
ENV PYTHONUNBUFFERED=1

# Set Lambda handler
CMD ["lambda.handler"] 