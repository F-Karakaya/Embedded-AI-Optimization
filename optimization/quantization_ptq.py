import torch
import torch.quantization
import torchvision.models as models
import os
import time

def main():
    print("Loading pretrained MobileNetV2...")
    # Load a pretrained MobileNetV2 model
    # Weights=MobileNet_V2_Weights.DEFAULT ensures we get the best available weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # Save original model size
    original_path = "models/original/mobilenet_v2.pt"
    torch.save(model.state_dict(), original_path)
    original_size = os.path.getsize(original_path) / (1024 * 1024)
    print(f"Original model saved to {original_path} ({original_size:.2f} MB)")

    print("Applying Post-Training Quantization (Dynamic)...")
    # Dynamic quantization is the simplest form, good for LSTM/Transformer, 
    # but for CNNs Static is usually preferred. However, purely for demonstration 
    # of the workflow on Windows without a calibration dataset, we will start with Dynamic 
    # or simulate a simple Static calibration if possible. 
    # Let's do Dynamic for simplicity and stability in this standalone script, 
    # but ideally we would do Static with a calibration loader.
    
    # Target linear layers for dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save quantized model
    quantized_path = "models/quantized/mobilenet_v2_ptq_dynamic.pt"
    torch.save(quantized_model.state_dict(), quantized_path)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"Quantized (Dynamic) model saved to {quantized_path} ({quantized_size:.2f} MB)")
    print(f"Reduction: {original_size / quantized_size:.2f}x")

    # --- Static Quantization Simulation ---
    # To demonstrate knowledge of static quantization (fusing, observing, calibrating)
    print("\nPreparing for Static Quantization (Simulation)...")
    
    # 1. Fuse (Conv+BN+Relu) - MobileNetV2 has specific patterns
    # For a real implementation we would fuse modules here.
    # model.fuse_model() # Hypothetical helper
    
    # 2. Config
    model_static = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model_static.eval()
    model_static.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 3. Prepare
    model_static = torch.quantization.prepare(model_static)
    
    # 4. Calibrate (Simulated with random data)
    print("Calibrating with dummy data...")
    input_shape = (1, 3, 224, 224)
    for _ in range(10):
        dummy_input = torch.randn(input_shape)
        model_static(dummy_input)
        
    # 5. Convert
    model_static = torch.quantization.convert(model_static)
    
    static_path = "models/quantized/mobilenet_v2_ptq_static.pt"
    torch.save(model_static.state_dict(), static_path)
    static_size = os.path.getsize(static_path) / (1024 * 1024)
    print(f"Quantized (Static) model saved to {static_path} ({static_size:.2f} MB)")

if __name__ == "__main__":
    os.makedirs("models/original", exist_ok=True)
    os.makedirs("models/quantized", exist_ok=True)
    main()
