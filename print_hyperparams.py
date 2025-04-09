#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
import torch
import json
from collections import OrderedDict

# Import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import get_args_parser
from build_modules import build_model, build_criterion, build_optimizer
from utils import resume_and_load


def format_section_header(title):
    """Format section header with decorative elements"""
    return f"\n{'=' * 80}\n{title}\n{'-' * 80}"


def print_model_architecture(args, model):
    """Print model architecture details"""
    print(format_section_header("MODEL ARCHITECTURE"))
    
    # Basic model configuration
    print(f"Backbone: {args.backbone}")
    print(f"Position Encoding: {args.pos_encoding}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Number of Queries: {args.num_queries}")
    print(f"Number of Feature Levels: {args.num_feature_levels}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    
    # Transformer details
    print(f"\nTransformer Configuration:")
    print(f"  - Number of Encoder Layers: {args.num_encoder_layers}")
    print(f"  - Number of Decoder Layers: {args.num_decoder_layers}")
    print(f"  - Number of Heads: {args.num_heads}")
    print(f"  - Feedforward Dimension: {args.feedforward_dim}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - With Box Refine: {'Yes' if args.with_box_refine else 'No'}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Size:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")


def print_optimization_params(args, optimizer):
    """Print optimization parameters"""
    print(format_section_header("OPTIMIZATION PARAMETERS"))
    
    # Learning rates
    print(f"Learning Rate: {args.lr}")
    print(f"Learning Rate Backbone: {args.lr_backbone}")
    print(f"Learning Rate Linear Proj: {args.lr_linear_proj}")
    
    # Learning rate scheduler
    print(f"\nLearning Rate Scheduler:")
    print(f"  - Type: StepLR")
    print(f"  - Step Size: {args.epoch_lr_drop}")
    print(f"  - Gamma: 0.1 (default)")
    
    # Optimizer details
    print(f"\nOptimizer:")
    print(f"  - Type: {'SGD' if args.sgd else 'AdamW'}")
    print(f"  - Weight Decay: {args.weight_decay}")
    if args.sgd:
        print(f"  - Momentum: 0.9")
    
    # Other training parameters
    print(f"\nTraining Configuration:")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Evaluation Batch Size: {args.eval_batch_size}")
    print(f"  - Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  - Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  - Number of Epochs: {args.epoch}")
    print(f"  - Gradient Clipping Max Norm: {args.clip_max_norm}")


def print_loss_coefficients(args, criterion):
    """Print loss coefficients"""
    print(format_section_header("LOSS COEFFICIENTS"))
    
    # General loss settings
    print(f"Only Class Loss: {'Yes' if args.only_class_loss else 'No'}")
    print(f"High Quality Matches: {'Yes' if args.high_quality_matches else 'No'}")
    
    # Loss coefficients
    print(f"\nLoss Weights:")
    print(f"  - Classification Loss Weight: {args.coef_class}")
    print(f"  - Bounding Box Loss Weight: {args.coef_boxes}")
    print(f"  - GIoU Loss Weight: {args.coef_giou}")
    print(f"  - Focal Loss Alpha: {args.alpha_focal}")
    print(f"  - EMA Alpha: {args.alpha_ema}")


def print_dataset_params(args):
    """Print dataset parameters"""
    print(format_section_header("DATASET PARAMETERS"))
    
    print(f"Data Root: {args.data_root}")
    print(f"Source Dataset: {args.source_dataset}")
    print(f"Target Dataset: {args.target_dataset}")


def print_training_mode_params(args):
    """Print training mode specific parameters"""
    print(format_section_header("TRAINING MODE PARAMETERS"))
    
    print(f"Training Mode: {args.mode}")
    
    # Mode-specific parameters
    if args.mode in ["teaching_standard", "teaching_mask"]:
        print(f"\nTeaching Parameters:")
        print(f"  - Keep Modules: {', '.join(args.keep_modules)}")
        print(f"  - Threshold: {args.threshold}")
        print(f"  - Alpha DT: {args.alpha_dt}")
        print(f"  - Gamma DT: {args.gamma_dt}")
        print(f"  - Max DT: {args.max_dt}")
        print(f"  - Dynamic Update: {'Yes' if args.dynamic_update else 'No'}")
        print(f"  - Fix Update Iteration: {args.fix_update_iter}")
        print(f"  - Max Update Iteration: {args.max_update_iter}")
        print(f"  - Use Pseudo Label Weights: {'Yes' if args.use_pseudo_label_weights else 'No'}")
        print(f"  - Use Loss Student: {'Yes' if args.use_loss_student else 'No'}")
    
    if args.mode == "teaching_mask":
        print(f"\nMasking Parameters:")
        print(f"  - Block Size: {args.block_size}")
        print(f"  - Masked Ratio: {args.masked_ratio}")
        print(f"  - Coefficient Masked Image: {args.coef_masked_img}")


def print_other_params(args):
    """Print other parameters"""
    print(format_section_header("OTHER PARAMETERS"))
    
    print(f"Device: {args.device}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Random Seed: {args.random_seed}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Print Frequency: {args.print_freq}")
    
    if args.resume:
        print(f"Resume from: {args.resume}")


def export_hyperparams_to_json(args, output_file):
    """Export all hyperparameters to a JSON file"""
    hyperparams = OrderedDict()
    
    # Convert args to dictionary and organize by categories
    args_dict = vars(args)
    
    # Model architecture
    hyperparams["model_architecture"] = {
        "backbone": args.backbone,
        "pos_encoding": args.pos_encoding,
        "num_classes": args.num_classes,
        "num_queries": args.num_queries,
        "num_feature_levels": args.num_feature_levels,
        "with_box_refine": args.with_box_refine,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout
    }
    
    # Optimization parameters
    hyperparams["optimization"] = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "lr_linear_proj": args.lr_linear_proj,
        "sgd": args.sgd,
        "weight_decay": args.weight_decay,
        "clip_max_norm": args.clip_max_norm,
        "epoch": args.epoch,
        "epoch_lr_drop": args.epoch_lr_drop
    }
    
    # Loss coefficients
    hyperparams["loss"] = {
        "only_class_loss": args.only_class_loss,
        "high_quality_matches": args.high_quality_matches,
        "coef_class": args.coef_class,
        "coef_boxes": args.coef_boxes,
        "coef_giou": args.coef_giou,
        "alpha_focal": args.alpha_focal,
        "alpha_ema": args.alpha_ema
    }
    
    # Dataset parameters
    hyperparams["dataset"] = {
        "data_root": args.data_root,
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset
    }
    
    # Mode specific parameters
    hyperparams["mode"] = {
        "mode": args.mode,
        "keep_modules": args.keep_modules,
        "threshold": args.threshold,
        "alpha_dt": args.alpha_dt,
        "gamma_dt": args.gamma_dt,
        "max_dt": args.max_dt,
        "dynamic_update": args.dynamic_update,
        "fix_update_iter": args.fix_update_iter,
        "max_update_iter": args.max_update_iter,
        "use_pseudo_label_weights": args.use_pseudo_label_weights,
        "use_loss_student": args.use_loss_student,
        "block_size": args.block_size,
        "masked_ratio": args.masked_ratio,
        "coef_masked_img": args.coef_masked_img
    }
    
    # Other parameters
    hyperparams["other"] = {
        "device": args.device,
        "output_dir": args.output_dir,
        "random_seed": args.random_seed,
        "num_workers": args.num_workers,
        "print_freq": args.print_freq,
        "resume": args.resume if args.resume else ""
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"\nHyperparameters exported to {output_file}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser('Print Model Hyperparameters', add_help=False)
    get_args_parser(parser)
    parser.add_argument('--export_json', default='', type=str, help='Export hyperparameters to JSON file')
    parser.add_argument('--compare_config', default='', type=str, help='Compare with another config JSON file')
    args = parser.parse_args()
    
    # Build model, criterion and optimizer for inspection
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    criterion = build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    
    # Load model weights if resume is specified
    if args.resume:
        model = resume_and_load(model, args.resume, device)
    
    # Print all hyperparameters
    print("\n" + "=" * 80)
    print(f"{'HYPERPARAMETERS REPORT':^80}")
    print("=" * 80)
    
    print_model_architecture(args, model)
    print_optimization_params(args, optimizer)
    print_loss_coefficients(args, criterion)
    print_dataset_params(args)
    print_training_mode_params(args)
    print_other_params(args)
    
    # Export hyperparameters to JSON if requested
    if args.export_json:
        export_hyperparams_to_json(args, args.export_json)
    
    # Compare with another config if requested
    if args.compare_config and os.path.exists(args.compare_config):
        print(format_section_header("CONFIGURATION COMPARISON"))
        try:
            with open(args.compare_config, 'r') as f:
                compare_config = json.load(f)
            
            current_config = vars(args)
            
            # Flatten the comparison config
            flat_compare = {}
            for category in compare_config:
                for key, value in compare_config[category].items():
                    flat_compare[key] = value
            
            # Find differences
            differences = []
            for key, value in current_config.items():
                if key in flat_compare and flat_compare[key] != value:
                    differences.append((key, value, flat_compare[key]))
            
            if differences:
                print("The following parameters differ from the comparison config:")
                for key, current_value, compare_value in differences:
                    print(f"  - {key}: Current={current_value}, Compare={compare_value}")
            else:
                print("No differences found between configurations.")
                
        except Exception as e:
            print(f"Error comparing configs: {str(e)}")


if __name__ == '__main__':
    main()