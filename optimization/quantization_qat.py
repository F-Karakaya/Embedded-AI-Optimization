import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os

def main():
    print("Setting up Quantization Aware Training (QAT)...")
    
    # 1. Load Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.train() # QAT requires training mode
    
    # 2. Fuse Modules
    # Ideally fuse Conv+BN+Relu. In MobileNetV2 implementation, this usually requires 
    # iterating over the features and identifying fuse-able blocks.
    # For this demo, we skip explicit fusion to keep the script robust against version changes,
    # or we can assume a simplified fusion for specific layers.
    model.fuse_model = lambda: None # Placeholder for actual fusion logic
    # model.fuse_model()
    
    # 3. Assign QConfig
    # 'qnnpack' is often used for mobile (ARM), 'fbgemm' for x86 server
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 4. Prepare
    model = torch.quantization.prepare_qat(model)
    
    # 5. Training Loop (Simulated)
    print("Starting QAT Fine-tuning (Simulated for 1 epoch/5 steps)...")
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    
    for step in range(5):
        # Dummy inputs and targets
        inputs = torch.randn(8, 3, 224, 224)
        targets = torch.randint(0, 1000, (8,))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}: Loss = {loss.item():.4f}")
        
    # 6. Convert
    model.eval()
    quantized_model = torch.quantization.convert(model)
    
    output_path = "models/quantized/mobilenet_v2_qat.pt"
    torch.save(quantized_model.state_dict(), output_path)
    print(f"QAT model saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("models/quantized", exist_ok=True)
    main()
