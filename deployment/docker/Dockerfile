# Professional Edge AI Deployment Container
# Base Image: Python Slim or specialized Manufacturers base (e.g. nvcr.io/nvidia/l4t-ml for Jetson)
# Using generic python-slim for Windows/Cross-platform compatibility demo.

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# libgl1-mesa-glx for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Model Artifacts
COPY models/quantized/mobilenet_v2_dynamic.onnx /app/model.onnx

# Copy Application Code
COPY edge_inference /app/edge_inference
COPY deployment /app/deployment

# Set Entrypoint to Realtime Pipeline
ENV PYTHONUNBUFFERED=1
CMD ["python", "edge_inference/realtime_pipeline.py"]
