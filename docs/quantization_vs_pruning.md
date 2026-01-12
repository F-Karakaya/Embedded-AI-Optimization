# Quantization vs. Pruning: An Engineering Perspective

## Quantization
**Definition**: The process of mapping continuous infinite values (32-bit floating point) to a smaller set of discrete finite values (e.g., 8-bit integers).

**Impact on Edge Hardware**:
- **Memory Footprint**: Direct 4x reduction (32-bit -> 8-bit).
- **Bandwidth**: Reduces memory bandwidth pressure, often the bottleneck on Jetson/Pi devices.
- **Compute**: Enables use of specialized integer arithmetic units (e.g., ARM NEON, NVIDIA Tensor Cores INT8 mode) which are significantly faster than FP32 pipelines.

**Techniques Demonstrated**:
1. **PTQ (Post-Training Quantization)**: Quick, requires no retraining. Good for standard CNNs. We demonstrated Dynamic and Static approaches.
2. **QAT (Quantization-Aware Training)**: Simulates quantization noise during forward pass, allowing weights to adapt. Essential for lower precision (INT4) or sensitive architectures (MobileNetV2 can be sensitive).

## Pruning
**Definition**: Removing unnecessary connections (weights) or neurons/channels from the network.

**Types**:
1. **Unstructured Pruning**: Zeroing out individual weights.
   - *Pros*: High sparsity potential (90%+).
   - *Cons*: **Hardware inefficient**. Standard GPUs/CPUs cannot skip zeros efficiently without sparse matrix library support, so file size shrinks (compression) but latency often remains unchanged.
2. **Structured Pruning**: Removing entire channels or filters.
   - *Pros*: Directly reduces FLOPs and matrix dimensions. "Real" speedup on all hardware.
   - *Cons*: Can severely degrade accuracy if not carefully fine-tuned.

## Strategic Selection
For this portfolio, we prioritize **INT8 Quantization** as the first-pass optimization for edge deployment because it offers guaranteed speedups on modern hardware (TFLite/TRT/ONNX) without architecture changes. Pruning is explored as a secondary optimization for constrained storage scenarios.
