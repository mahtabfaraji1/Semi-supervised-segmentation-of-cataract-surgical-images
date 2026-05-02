"""
Test script for the Mean Teacher semi-supervised segmentation model.
Reference:
    Faraji et al. "An Evaluation of the Mean Teacher Framework for
    Semi-Supervised Cataract Surgical Image Segmentation."
    Transl Vis Sci Technol. 2026;15(4):5.
    https://doi.org/10.1167/tvst.15.4.5

    Usage:
    python test_mt.py \
        --image_dir ./data/test/images \
        --label_dir ./data/test/labels \
        --model_path ./model/unet_resnet_best_model.pth \
        --output_dir ./results/mt_100labeled \
        --num_classes 5


"""

import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from networks.net_factory import net_factory
from test_utils import run_inference, aggregate_and_print


# Model loading

def load_student_model(args, device):

    net = net_factory(
        net_type=args.model,
        in_chns=3,
        class_num=args.num_classes,
        backbone=args.model_backbone
    )
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    net.eval()
    print(f"Loaded student model weights from: {args.model_path}")
    return net


# Inference loop

def run_test(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    net = load_student_model(args, device)

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

    summary = aggregate_and_print(metric_dict, args.num_classes)

    pkl_name = (
        f"metrics_mt"
        f"_labeled={args.labeled_num}"
        f"_unlabeled={args.unlabeled_num}"
        f"_classes={args.num_classes}.pkl"
    )
    pkl_path = os.path.join(args.output_dir, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(metric_dict, f)
    return metric_dict, summary


# Argument parsing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Mean Teacher semi-supervised model"
    )

    # Paths
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing test PNG images')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing ground-truth label PNGs '
                             '(grayscale, pixel = class index)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved  model checkpoint (.pth). '
                             'train_mt.py saves this as: '
                             '<snapshot_dir>/<model>_best_model.pth')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predictions and metrics')

    # Model
    parser.add_argument('--model', type=str, default='unet_resnet',
                        help='Network architecture.')
    parser.add_argument('--model_backbone', type=str, default='resnet50',
                        help='Encoder backbone.')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes including background. '
                             '5 (background, iris, pupil, IOL, instrument). '
                             )
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256],
                        help='Inference input size [H W].')

    # Experiment metadata (used in output filename only)
    parser.add_argument('--labeled_num', type=int, default=100,
                        help='Number of labeled images used in training. '
                             )
    parser.add_argument('--unlabeled_num', type=int, default=40000,
                        help='Number of unlabeled images used in training. '
                             )

    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index')

    args = parser.parse_args()
    run_test(args)
