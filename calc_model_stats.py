import os
import argparse
import torch
import numpy as np
from torch import nn

from models.deformable_transformer import DeformableTransformer
from models.deformable_detr import DeformableDETR
from models.positional_encoding import PositionEncodingSine
from models.backbones import ResNet18MultiScale, ResNet50MultiScale, ResNet101MultiScale

def calculate_parameters(model):
    """Calculate number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(backbone_name, num_classes=9):
    """Create model based on backbone name"""
    if backbone_name == 'resnet18':
        backbone = ResNet18MultiScale()
    elif backbone_name == 'resnet50':
        backbone = ResNet50MultiScale()
    elif backbone_name == 'resnet101':
        backbone = ResNet101MultiScale()
    else:
        raise ValueError(f'Invalid backbone name: {backbone_name}')
    
    position_encoding = PositionEncodingSine()
    transformer = DeformableTransformer(
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        feedforward_dim=1024,
        dropout=0.1
    )
    
    model = DeformableDETR(
        backbone=backbone,
        position_encoding=position_encoding,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=300,
        num_feature_levels=4
    )
    return model

def estimate_flops(model, input_size=(1, 3, 1024, 2048)):
    """
    Estimate FLOPs for the model with the given input size
    This is a rough estimation based on common operations
    """
    h, w = input_size[2], input_size[3]
    
    # Rough estimate for ResNet50 with the input size
    if isinstance(model.backbone, ResNet50MultiScale):
        backbone_flops = 4.1e9 * (h * w) / (224 * 224)  # Scale from ImageNet size (224x224)
    elif isinstance(model.backbone, ResNet18MultiScale):
        backbone_flops = 1.8e9 * (h * w) / (224 * 224) 
    elif isinstance(model.backbone, ResNet101MultiScale):
        backbone_flops = 7.6e9 * (h * w) / (224 * 224)
    else:
        backbone_flops = 4.0e9 * (h * w) / (224 * 224)  # Default estimate
    
    # Transformer FLOPs estimation
    # Based on rough estimates for attention and FFN operations
    transformer = model.transformer
    d_model = transformer.hidden_dim
    n_heads = transformer.num_heads
    
    # Estimate for feature maps sizes after backbone
    feature_h, feature_w = h//32, w//32  # Approximation for the smallest feature map
    
    # Encoder complexity
    num_encoder_elements = feature_h * feature_w * 1.5  # Accounting for multi-scale features
    # Get encoder layers count from encoder module
    encoder_layers_count = len(transformer.encoder.layers)
    encoder_attention_flops = encoder_layers_count * num_encoder_elements * (
        # Self-attention
        4 * d_model * d_model +  # Projections
        2 * num_encoder_elements * d_model  # Attention weights
    )
    encoder_ffn_flops = encoder_layers_count * num_encoder_elements * (
        4 * d_model * transformer.feedforward_dim
    )
    
    # Decoder complexity
    num_queries = model.num_queries
    # Get decoder layers count from decoder module
    decoder_layers_count = len(transformer.decoder.layers)
    decoder_attention_flops = decoder_layers_count * (
        # Self-attention among queries
        4 * d_model * d_model * num_queries +
        # Cross-attention
        4 * d_model * d_model * num_queries +
        2 * num_queries * num_encoder_elements * d_model
    )
    decoder_ffn_flops = decoder_layers_count * num_queries * (
        4 * d_model * transformer.feedforward_dim
    )
    
    # Projection and head flops
    projection_flops = d_model * sum(ch * d_model for ch in model.backbone.num_channels)
    head_flops = num_queries * (d_model * model.num_classes + 4 * d_model * 3)
    
    # Total model flops
    total_flops = (backbone_flops + encoder_attention_flops + encoder_ffn_flops + 
                   decoder_attention_flops + decoder_ffn_flops + projection_flops + head_flops)
    
    return total_flops

def analyze_model(backbone_name='resnet50', num_classes=9, input_size=(1, 3, 1024, 2048)):
    """Analyze model and print statistics"""
    try:
        model = create_model(backbone_name, num_classes)
        model.eval()
        
        # Calculate total parameters
        total_params = calculate_parameters(model)
        
        # Estimate FLOPs
        flops = estimate_flops(model, input_size)
        
        # Convert to more readable format
        if flops > 1e12:
            flops_str = f"{flops / 1e12:.2f} TFLOPs"
        else:
            flops_str = f"{flops / 1e9:.2f} GFLOPs"
        
        print(f"Backbone: {backbone_name}")
        print(f"Input size: {input_size[2]}x{input_size[3]}")
        print(f"Number of trainable parameters: {total_params:,}")
        print(f"Estimated computational complexity: {flops_str}")
        
        # Breakdown parameters by components
        backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
        other_params = total_params - backbone_params - transformer_params
        
        print("\nParameter distribution:")
        print(f"  Backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
        print(f"  Transformer: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
        print(f"  Other (projection, heads, etc.): {other_params:,} ({other_params/total_params*100:.1f}%)")
        
        return {
            "backbone": backbone_name,
            "total_params": total_params,
            "flops": flops,
            "backbone_params": backbone_params,
            "transformer_params": transformer_params,
            "other_params": other_params
        }
        
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_training_scenarios(backbone_name='resnet50', num_classes=9, input_size=(1, 3, 1024, 2048)):
    """Compare different training scenarios"""
    print("=" * 80)
    print(f"Comparing Training Scenarios with {backbone_name} backbone")
    print("=" * 80)
    
    # All scenarios use the same model architecture - only training strategy differs
    stats = analyze_model(backbone_name, num_classes, input_size)
    
    if stats is None:
        print("Failed to analyze model. Cannot compare scenarios.")
        return
    
    print("\nComparison between training scenarios:")
    
    # Source-only scenario
    print("\n1. Source-Only Scenario:")
    print("   - Uses a single model trained on source domain")
    print(f"   - Parameters: {stats['total_params']:,}")
    print(f"   - FLOPs: {stats['flops']/1e9:.2f} GFLOPs")
    
    # Teaching_standard scenario (Mean Teacher)
    print("\n2. Teaching-Standard Scenario (MT):")
    print("   - Uses student and teacher models (same architecture)")
    print("   - Teacher generates pseudo labels, student learns from them")
    print("   - Teacher is updated via EMA from the student")
    print("   - Inference-time parameters identical to source-only")
    print("   - Training-time parameters (both models):")
    print(f"     * Parameters: {stats['total_params']*2:,} (2x source-only)")
    print(f"     * FLOPs for forward pass: {stats['flops']/1e9*2:.2f} GFLOPs (2x source-only)")
    print("   - Training-time gradient updates: 1x (only student model)")
    
    # Teaching_mask scenario (proposed DRU)
    print("\n3. Teaching-Mask Scenario (DRU):")
    print("   - Uses three models with the same architecture:")
    print("     * Student model: Actively trained with gradients")
    print("     * Teacher model: Updated via EMA, no gradients")
    print("     * Initial student model: Historical snapshot, no gradients")
    print("   - Teacher generates primary pseudo labels")
    print("   - Initial student model provides additional supervision signals without being trained")
    print("   - Student is trained with masked images and multiple supervision signals")
    print("   - Inference-time parameters identical to source-only")
    print("   - Training-time memory footprint (all three models):")
    print(f"     * Parameters: {stats['total_params']*3:,} (3x source-only)")
    print(f"     * FLOPs for forward pass: {stats['flops']/1e9*3:.2f} GFLOPs (3x forward passes)")
    print("   - Training-time gradient updates: 1x (only student model)")
    
    print("\nNotes:")
    print("- FLOPs are estimated for a single forward pass with input size {0}x{1}".format(input_size[2], input_size[3]))
    print("- These are approximations meant for relative comparison between scenarios")
    print("- At inference time, all scenarios use only one model, so inference complexity is identical")
    print("- While teaching_mask uses 3 models in memory, only the student model receives gradient updates")

def main():
    parser = argparse.ArgumentParser(description='Calculate model parameters and FLOPs')
    parser.add_argument('--backbone', default='resnet50', choices=['resnet18', 'resnet50', 'resnet101'],
                        help='Backbone model (default: resnet50)')
    parser.add_argument('--classes', default=9, type=int, help='Number of classes (default: 9)')
    parser.add_argument('--height', default=1024, type=int, help='Input height (default: 1024)')
    parser.add_argument('--width', default=2048, type=int, help='Input width (default: 2048)')
    parser.add_argument('--compare', action='store_true', help='Compare training scenarios')
    
    args = parser.parse_args()
    
    input_size = (1, 3, args.height, args.width)
    
    if args.compare:
        compare_training_scenarios(args.backbone, args.classes, input_size)
    else:
        print("=" * 80)
        print(f"Model Statistics for {args.backbone} with input {args.height}x{args.width}")
        print("=" * 80)
        analyze_model(args.backbone, args.classes, input_size)

if __name__ == '__main__':
    main()