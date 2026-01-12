import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import os

def main():
    print("Loading model for Pruning...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    print("Applying Global Unstructured Pruning (L1 Norm)...")
    
    # Identify parameters to prune (e.g., all Conv2d weights)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
            
    if not parameters_to_prune:
        print("No Conv2d layers found? Check model architecture.")
        return

    print(f"Pruning {len(parameters_to_prune)} Conv2d layers with 20% sparsity...")
    
    # prune.global_unstructured applies pruning across the whole model
    # The 20% smallest weights globally will be zeroed out
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    
    # Remove the pruning re-parameterization to make the pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, 'weight')
        
    output_path = "models/pruned/mobilenet_v2_pruned_unstructured.pt"
    torch.save(model.state_dict(), output_path)
    
    print(f"Pruned model saved to {output_path}")
    print("Note: Unstructured pruning requires sparse tensor hardware support to realize speedups.")
    print("For general hardware, structured pruning or 2:4 sparsity is preferred.")
    
    # Example of Structured Pruning (Pruning entire channels)
    print("\nApplying Structured Pruning Example (Single Layer)...")
    # Reload fresh model
    model_struct = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    conv_layer = model_struct.features[0][0] # First conv layer
    
    # Prune 50% of channels along dim 0 (output channels) based on L1 norm
    prune.ln_structured(conv_layer, name="weight", amount=0.5, n=1, dim=0)
    prune.remove(conv_layer, 'weight')
    
    print("First Conv2d layer pruned (Structured 50%).")
    # We won't save this one, just demonstrating the API usage for the recruiter.

if __name__ == "__main__":
    os.makedirs("models/pruned", exist_ok=True)
    main()
