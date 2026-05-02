"""
Test script for fully supervised segmentation models (UNet-ResNet50, SwinUNet).

Reference:
    Faraji et al. "An Evaluation of the Mean Teacher Framework for
    Semi-Supervised Cataract Surgical Image Segmentation."
    Transl Vis Sci Technol. 2026;15(4):5.
    https://doi.org/10.1167/tvst.15.4.5

Usage:
    # UNet-ResNet50
    python test_supervised.py \
        --model unet_resnet \
        --image_dir ./data/test/images \
        --label_dir ./data/test/labels \
        --model_path ./model/unet_resnet_best_model.pth \
        --output_dir ./results/unet_resnet \
        --num_classes 5

    # SwinUNet
    python test_supervised.py \
        --model swinunet \
        --image_dir ./data/test/images \
        --label_dir ./data/test/labels \
        --model_path ./model/swinunet_best_model.pth \
        --output_dir ./results/swinunet \
        --num_classes 5 \
        --patch_size 224 224 \
        --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
"""

import argparse
import os
import pickle
import torch
from tqdm import tqdm

from test_utils import run_inference, aggregate_and_print


# Model loading

def load_model(args, device):
    """
    Load the trained supervised model from checkpoint.

    UNet-ResNet50: loaded via net_factory.
    SwinUNet:      loaded via vision_transformer with Swin config.

    Args:
        args:   Parsed arguments.
        device: torch.device.

    Returns:
        torch.nn.Module: Model in eval mode on device.
    """
    if args.model == "swinunet":
        from config import get_config
        from networks.vision_transformer import SwinUnet as ViT_seg
        swin_config = get_config(args)
        net = ViT_seg(swin_config, img_size=args.patch_size, num_classes=args.num_classes)
    else:
        from networks.net_factory import net_factory
        net = net_factory(
            net_type=args.model,
            in_chns=3,
            class_num=args.num_classes,
            backbone=args.model_backbone
        )

    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    net.eval()
    print(f"Loaded {args.model} weights from: {args.model_path}")
    return net


# Inference loop

def run_test(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    net = load_model(args, device)

    # Collect all test image paths
    image_files = sorted([
        f for f in os.listdir(args.image_dir) if f.endswith('.png')
    ])
    print(f"Found {len(image_files)} test images in {args.image_dir}")

    metric_dict = {cls_id: [] for cls_id in range(1, args.num_classes)}

    for image_file in tqdm(image_files, desc="Testing"):
        image_path = os.path.join(args.image_dir, image_file)
        label_path = os.path.join(args.label_dir, image_file)

        if not os.path.exists(label_path):
            print(f"Warning: label not found for {image_file}, skipping.")
            continue

        metrics_per_class = run_inference(
            image_path, label_path, net,
            args.output_dir, args, device
        )

        for cls_id, vals in metrics_per_class.items():
            if cls_id in metric_dict:
                metric_dict[cls_id].append(vals)

    # Print results table
    summary = aggregate_and_print(metric_dict, args.num_classes)

    # Save raw metric dict for statistical testing
    pkl_name = f"metrics_{args.model}_supervised.pkl"
    pkl_path = os.path.join(args.output_dir, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(metric_dict, f)
    print(f"Raw metrics saved to: {pkl_path}")

    return metric_dict, summary


# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test supervised segmentation models -- UNet-ResNet50 or SwinUNet"
    )

    # Paths
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing test PNG images')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing ground-truth label PNGs '
                             '(grayscale, pixel = class index)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint (.pth). '
                             'Paper: <snapshot_dir>/<model>_best_model.pth')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predictions and metrics')

    # Model
    parser.add_argument('--model', type=str, default='unet_resnet',
                        choices=['unet_resnet', 'swinunet'],
                        help='Model architecture.')
    parser.add_argument('--model_backbone', type=str, default='resnet50',
                        help='Backbone for unet_resnet.')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes including background. '
                             '5 (background, iris, pupil, IOL, instrument)')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256],
                        help='Inference input size [H W]. '
                             '256 256 (UNet), 224 224 (SwinUNet)')

    # Hardware
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index')

    # SwinUNet-specific (only used when --model swinunet)
    swin_group = parser.add_argument_group('SwinUNet', 'Only used when --model swinunet')
    swin_group.add_argument('--cfg', type=str,
                            default='./configs/swin_tiny_patch4_window7_224_lite.yaml',
                            help='Path to Swin config YAML')
    swin_group.add_argument('--opts', nargs='+', default=None)
    swin_group.add_argument('--zip', action='store_true')
    swin_group.add_argument('--cache-mode', type=str, default='part',
                            choices=['no', 'full', 'part'])
    swin_group.add_argument('--resume', default=None)
    swin_group.add_argument('--accumulation-steps', type=int, default=None)
    swin_group.add_argument('--use-checkpoint', action='store_true')
    swin_group.add_argument('--amp-opt-level', type=str, default='O1',
                            choices=['O0', 'O1', 'O2'])
    swin_group.add_argument('--tag', default=None)
    swin_group.add_argument('--eval', action='store_true')
    swin_group.add_argument('--throughput', action='store_true')

    args = parser.parse_args()
    run_test(args)
