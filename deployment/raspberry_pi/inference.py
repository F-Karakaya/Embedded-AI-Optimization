import time
import argparse
import numpy as np
import cv2
import os

# NOTE: This script is designed for Raspberry Pi (ARM64).
# It uses onnxruntime for inference (efficient on ARM CPU).
# To simulate on Windows, we ensure dependencies are cross-platform.

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found. Please install via: pip install onnxruntime")

def load_image(path, size=(224, 224)):
    # Standard preprocessing
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Inference Optimization Demo')
    parser.add_argument('--model', type=str, default='../../models/quantized/mobilenet_v2_dynamic.onnx', help='Path to ONNX model')
    args = parser.parse_args()

    print(f"[Pi] Loading model: {args.model}")
    
    # Session Options for ARM CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Assuming Pi 4 with 4 cores
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(args.model, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    input_name = session.get_inputs()[0].name
    
    # Dummy Input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    # Latency Test
    print("Measuring Latency over 50 iterations...")
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        timings.append((end - start) * 1000)

    avg_latency = np.mean(timings)
    print(f"[Pi Benchmark] Average Inference Latency: {avg_latency:.2f} ms")
    print(f"[Pi Benchmark] Estimated FPS: {1000/avg_latency:.2f}")

if __name__ == "__main__":
    main()
