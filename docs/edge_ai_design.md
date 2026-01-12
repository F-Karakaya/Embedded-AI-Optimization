# Edge AI System Design

## 1. The Inference Pipeline
Edge AI is not just the model `forward()`. It is a strict timeline:

`[Camera Capture] -> [Pre-process (Resize/Norm)] -> [Transfer to Device] -> [Inference] -> [Transfer to Host] -> [Post-process (NMS/Argmax)] -> [Action]`

### Design Constraints
- **Latency Budget**: For 30 FPS, total time < 33ms.
- **Jitter**: Variance in inference time must be low for control systems (robotics).
- **Memory**: Model + runtime must fit in RAM. Pi 4 (4GB) is generous; microcontroller (256KB) is not.

## 2. Optimization Implementation
We chose **MobileNetV2** as the backbone.
- **Why?**: It uses Depthwise Separable Convolutions, significantly reducing FLOPs compared to ResNet.
- **Optimization**: We applied Post-Training Dynamic Quantization.
- **Result**: ~4x size reduction, ~2-3x CPU speedup.

## 3. Deployment Strategy
We containerize the application to ensure consistency.
- **Docker**: Encapsulates dependencies (OpenCV, ONNX Runtime).
- **Volume Mapping**: Access to camera device (`--device /dev/video0`) is handled at runtime.

This design ensures that the software developed on a workstation (Windows 10) behaves identically on the target edge device (Linux/ARM), barring hardware acceleration differences.
