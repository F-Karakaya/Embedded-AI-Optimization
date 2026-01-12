# Edge Accelerator Comparison

Choosing the right silicon is a trade-off between TOPS (Tera Operations Per Second), Power (Watts), and Form Factor.

| Platform | Processor | AI Accelerator | Peak Performance | Power | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raspberry Pi 4** | ARM Cortex-A72 (CPU) | None (CPU only) | ~5-10 GFLOPS | 3-5W | Low cost, simple logic, Educational |
| **Google Coral USB** | (Host CPU) | Edge TPU (ASIC) | 4 TOPS (INT8) | +2W | Adding AI to legacy/low-power hosts |
| **NVIDIA Jetson Nano** | ARM Cortex-A57 | Maxwell GPU (128 core) | 0.47 TFLOPS (FP16) | 5-10W | Entry-level vision, CUDA ecosystem |
| **NVIDIA Orin Nano** | ARM Cortex-A78AE | Ampere GPU (1024 core) | 20-40 TOPS (INT8) | 7-15W | Modern robotics, Transformer models |
| **Qualcomm/Android** | ARM Cortex | Hexagon DSP/NPU | 10-50+ TOPS | <5W | Mobile phones, extremely low power |

## Engineering Decision for this Project
Since we target a **General Purpose Edge Deployment**, we focus on **ONNX Runtime** and **TensorRT**.
- **ONNX**: Provides the broadest compatibility across ARM CPUs (Pi) and NPUs.
- **TensorRT**: The gold standard for NVIDIA embedded GPUs (Jetson), offering the best latency per watt.

By preparing our MobileNetV2 in **ONNX (INT8)**, we cover the majority of edge targets (Pi, Coral via compiler, generic ARM).
