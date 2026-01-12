import time
import numpy as np
import onnxruntime as ort
import os
import threading

def measure_throughput(model_path, duration=5.0, batch_size=1):
    if not os.path.exists(model_path):
        return
        
    print(f"Benchmarking Throughput (FPS): {os.path.basename(model_path)} [Batch {batch_size}]")
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    # Warmup
    session.run(None, {input_name: dummy_input})
    
    start_time = time.perf_counter()
    end_time = start_time + duration
    frames_processed = 0
    
    while time.perf_counter() < end_time:
        session.run(None, {input_name: dummy_input})
        frames_processed += batch_size
        
    total_time = time.perf_counter() - start_time
    fps = frames_processed / total_time
    
    print(f"  Total Frames: {frames_processed}")
    print(f"  Duration: {total_time:.2f}s")
    print(f"  FPS: {fps:.2f}")
    
    return fps

def main():
    print("--- Edge AI Throughput Benchmark ---")
    
    models_to_test = [
        "models/original/mobilenet_v2.onnx",
        "models/quantized/mobilenet_v2_dynamic.onnx"
    ]
    
    with open("benchmarking/fps_results.csv", "w") as f:
        f.write("Model,Batch Size,FPS\n")
        
        for m in models_to_test:
            if os.path.exists(m):
                # Single stream (Batch 1)
                fps_1 = measure_throughput(m, batch_size=1)
                f.write(f"{m},1,{fps_1:.2f}\n")
                
                # Batch 4 (Simulating accumulated input)
                fps_4 = measure_throughput(m, batch_size=4)
                f.write(f"{m},4,{fps_4:.2f}\n")

if __name__ == "__main__":
    main()
