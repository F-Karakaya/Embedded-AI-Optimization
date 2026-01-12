# Raspberry Pi Setup & Optimization Guide

## Hardware Constraints
- **Device**: Raspberry Pi 4 Model B (4GB/8GB RAM)
- **CPU**: Quad-core Cortex-A72 (ARM v8) @ 1.5GHz
- **Accelerator**: None (CPU Inference) or USB Accelerator (Coral TPU)
- **Power**: ~3-5W active load

## Environment Setup
Standard Raspbian (64-bit) recommended for best math performance.

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev libopencv-dev python3-opencv
pip3 install onnxruntime numpy
```

## Optimization Strategy
1. **Model Format**: Use ONNX or TensorFlow Lite. PyTorch (LibTorch) is heavy and often slower on bare Pi CPU.
2. **Quantization**: INT8 Quantization is critical. ARM NEON instructions handle INT8 efficiently.
3. **Threading**: Set `intra_op_num_threads` to match physical cores (4). Over-subscription hurts performance.
4. **Thermal**: Ensure active cooling. Thermal throttling drops CPU clock from 1.5GHz to 600MHz, destroying latency metrics.

## Execution
Run the inference script:
```bash
python3 inference.py --model mobilenet_v2_quantized.onnx
```
