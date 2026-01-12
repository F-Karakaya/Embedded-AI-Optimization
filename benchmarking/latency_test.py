import time
import numpy as np
import onnxruntime as ort
import os
import argparse

def measure_latency(model_path, runs=100, warmups=10):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Benchmarking Latency: {os.path.basename(model_path)}")
    
    # Load Session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Handle dynamic axes for dummy input
    input_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
    # Fallback if unknown
    if input_shape[1] is None: input_shape = (1, 3, 224, 224)
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmups):
        session.run(None, {input_name: dummy_input})
        
    # Timed Runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # ms
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"  Avg: {np.mean(latencies):.3f} ms")
    print(f"  P50: {p50:.3f} ms")
    print(f"  P95: {p95:.3f} ms")
    print(f"  P99: {p99:.3f} ms")
    
    return np.mean(latencies)

def main():
    print("--- Edge AI Latency Benchmark ---")
    print("Platform: Windows 10 (Simulated Edge Environment via CPU)")
    
    models_to_test = [
        "models/original/mobilenet_v2.onnx",
        "models/quantized/mobilenet_v2_dynamic.onnx"
    ]
    
    results = []
    for m in models_to_test:
        if os.path.exists(m):
            avg = measure_latency(m)
            results.append((m, avg))
    
    # Save results
    with open("benchmarking/latency_results.csv", "w") as f:
        f.write("Model,Average Latency (ms)\n")
        for name, lat in results:
            f.write(f"{name},{lat:.3f}\n")
            
    print("\nResults saved to benchmarking/latency_results.csv")

if __name__ == "__main__":
    main()
