import cv2
import numpy as np
import time
import argparse
import onnxruntime as ort
from camera_stream import CameraStream

def preprocess(frame, input_shape=(224, 224)):
    """
    Standard resize and normalize.
    """
    img = cv2.resize(frame, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../models/quantized/mobilenet_v2_dynamic.onnx')
    args = parser.parse_args()

    print("Initializing Edge AI Pipeline...")
    
    # 1. Load Model (Quantized ONNX)
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(args.model, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load model {args.model}: {e}")
        print("Please run optimization scripts first.")
        return

    input_name = session.get_inputs()[0].name
    
    # 2. Start Camera
    cam = CameraStream(src=0).start()
    print("Camera stream started. Waiting for frames...")
    
    # Wait for first frame
    while cam.read() is None:
        time.sleep(0.1)

    print("Starting Inference Loop. Press 'q' to exit.")
    
    fps_avg = 0
    alpha = 0.1
    
    try:
        while True:
            t_start = time.perf_counter()
            
            # A. Capture
            frame = cam.read()
            if frame is None: continue
            
            # B. Preprocess
            input_tensor = preprocess(frame)
            
            # C. Inference
            ort_inputs = {input_name: input_tensor}
            outputs = session.run(None, ort_inputs)
            
            # D. Postprocess (Argmax)
            # Typically logic to decode ImageNet class, we just show index here
            probs = outputs[0][0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] # Note: Softmax not strictly needed for argmax
            
            t_end = time.perf_counter()
            dt = t_end - t_start
            fps_inst = 1.0 / dt if dt > 0 else 0
            fps_avg = (alpha * fps_inst) + ((1-alpha) * fps_avg)
            
            # E. Visualize
            display = frame.copy()
            cv2.putText(display, f"FPS: {fps_avg:.1f} (Latency: {dt*1000:.1f}ms)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Class: {pred_idx}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Edge AI Pipeline", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
