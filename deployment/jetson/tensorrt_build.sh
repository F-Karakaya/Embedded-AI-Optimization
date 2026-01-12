#!/bin/bash

# TensorRT Engine Build Script
# Usage: ./tensorrt_build.sh <onnx_model_path> <output_engine_path>

ONNX_MODEL=$1
OUTPUT_ENGINE=$2
TRT_BIN="/usr/src/tensorrt/bin/trtexec"

if [ -z "$ONNX_MODEL" ]; then
    echo "Error: Please provide input ONNX model path."
    exit 1
fi

echo "Building TensorRT Engine from $ONNX_MODEL..."

# Explain flags:
# --fp16: Enable FP16 precision (critical for Jetson performance)
# --int8: Enable INT8 precision (requires calibration cache usually, simply enabling mixing here)
# --workspace: Allow GPU memory for tactics optimization
# --explicitBatch: Required for modern ONNX files

$TRT_BIN \
    --onnx=$ONNX_MODEL \
    --saveEngine=$OUTPUT_ENGINE \
    --fp16 \
    --workspace=1024 \
    --verbose \
    --explicitBatch

echo "Build complete. Engine saved to $OUTPUT_ENGINE"
