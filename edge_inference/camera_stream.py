import cv2
import numpy as np
import time
import threading

class CameraStream:
    """
    Simulates a camera stream or wraps a real webcam if available.
    Designed to run in a separate thread to prevent blocking the inference loop.
    """
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        # Try opening camera, fallback to synthetic
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[Camera] Warning: Could not open source {src}. Switching to Synthetic Mode.")
            self.mode = 'synthetic'
        else:
            self.mode = 'real'
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.mode == 'real':
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    # Loop video if file, or handle disconnect
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # Synthetic: Bouncing ball or noise
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Create moving visual to prove liveness
                t = time.time()
                cx = int(self.width/2 + 100 * np.sin(t*2))
                cy = int(self.height/2 + 100 * np.cos(t*2))
                cv2.circle(img, (cx, cy), 20, (0, 255, 0), -1)
                cv2.putText(img, f"Simulated Input {t:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                with self.lock:
                    self.frame = img
                time.sleep(0.033) # ~30 FPS limit

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.mode == 'real':
            self.cap.release()
