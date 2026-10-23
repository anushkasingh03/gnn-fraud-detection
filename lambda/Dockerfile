# AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Install ONNX Runtime dependencies
COPY lambda/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the inference handler
COPY lambda/lambda_handler.py ${LAMBDA_TASK_ROOT}/lambda_handler.py

# Copy the exported GraphSAGE model
COPY models/real_baseline_sage.onnx ${LAMBDA_TASK_ROOT}/models/real_baseline_sage.onnx

# Lambda entry point
CMD ["lambda_handler.handler"]