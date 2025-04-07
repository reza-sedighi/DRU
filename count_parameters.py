import argparse
import torch
from pathlib import Path

from build_modules import build_model
from utils.distributed_utils import init_distributed_mode

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Parse arguments using the same parser as in main.py
    parser = argparse.ArgumentParser('DRU Parameter Counter')
    
    # Model Settings (minimal required settings from main.py)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    
    # Parse args
    args = parser.parse_args()
    
    # Initialize distributed mode (minimal setup)
    args.distributed = False
    
    # Build device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Build model
    print("Building model with backbone:", args.backbone)
    model = build_model(args, device)
    
    # Count parameters
    params = count_parameters(model)
    print(f"Single model parameters: {params:,}")
    
    # For teaching scenarios:
    print("\nParameter count in different scenarios:")
    print(f"1. source_only: {params:,} parameters")
    print(f"2. teaching_standard: {2*params:,} parameters (student + teacher)")
    print(f"3. teaching_mask: {3*params:,} parameters (student + teacher + init_student)")
    
    # Memory usage estimation (rough estimate, varies by implementation)
    bytes_per_param = 4  # 32-bit float
    print("\nApproximate memory usage:")
    print(f"1. source_only: {params * bytes_per_param / (1024**2):.2f} MB")
    print(f"2. teaching_standard: {2 * params * bytes_per_param / (1024**2):.2f} MB")
    print(f"3. teaching_mask: {3 * params * bytes_per_param / (1024**2):.2f} MB")
    
    print("\nNote: Actual memory usage will be higher due to optimizer states, gradients, and intermediate activations.")
    
    # Breakdown by model components
    print("\nParameter breakdown by major components:")
    backbone_params = sum(p.numel() for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad)
    transformer_params = sum(p.numel() for name, p in model.named_parameters() if 'transformer' in name and p.requires_grad)
    others = params - backbone_params - transformer_params
    
    print(f"- Backbone: {backbone_params:,} parameters ({backbone_params/params*100:.2f}%)")
    print(f"- Transformer: {transformer_params:,} parameters ({transformer_params/params*100:.2f}%)")
    print(f"- Others: {others:,} parameters ({others/params*100:.2f}%)")

if __name__ == '__main__':
    main()