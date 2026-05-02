"""
Fully Supervised Training for Cataract Surgical Image Segmentation.

Supports two architectures evaluated in the paper as supervised baselines:
    - UNet with ResNet50 backbone  (train_supervised.py --model unet_resnet)
    - SwinUNet                     (train_supervised.py --model swinunet)

Reference:
    Faraji et al. "An Evaluation of the Mean Teacher Framework for
    Semi-Supervised Cataract Surgical Image Segmentation."
    Transl Vis Sci Technol. 2026;15(4):5.
    https://doi.org/10.1167/tvst.15.4.5

Usage:
    # UNet-ResNet50 baseline
    python train_fully_supervised.py \
        --model unet_resnet \
        --root_path ./data/ \
        --max_iterations 40000 \
        --batch_size 24 \
        --base_lr 0.01 \
        --patch_size 256 256

    # SwinUNet baseline
    python train_fully_supervised.py \
        --model swinunet \
        --root_path ./data/ \
        --max_iterations 40000 \
        --batch_size 24 \
        --base_lr 0.001 \
        --patch_size 224 224 \
        --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator
from utils import losses
from val import test_single_image

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.net_factory import net_factory

def worker_init_fn(worker_id):
    """Fix random seed per DataLoader worker for reproducibility."""
    random.seed(1337 + worker_id)


def build_model(args, device):
    """
    Instantiate the segmentation model specified by --model.

    UNet-ResNet50:  loaded via net_factory, ImageNet pretrained.
    SwinUNet:       loaded via vision_transformer, Swin-Tiny pretrained.
                    Requires --cfg pointing to the Swin config YAML.

    """
    if args.model == "swinunet":
        swin_config = get_config(args)
        model = ViT_seg(swin_config, img_size=args.patch_size, num_classes=args.num_classes)
    else:
        model = net_factory(
            net_type=args.model,
            in_chns=3,
            class_num=args.num_classes,
            backbone=args.model_backbone
        )

    model.to(device)
    return model


# Training

def train(args):
    num_classes    = args.num_classes
    max_iterations = args.max_iterations
    patch_size     = args.patch_size

    # Experiment name and snapshot directory
    exp_name = (
        f"supervised_{args.model}"
        f"_lr={args.base_lr}"
        f"_bs={args.batch_size}"
        f"_labeled={args.labeled_num}"
        f"_classes={num_classes}"
    )
    snapshot_path = os.path.join("../model", exp_name)
    os.makedirs(snapshot_path, exist_ok=True)

    # Logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = build_model(args, device)
    model.train()

    # Datasets
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    trainloader = DataLoader(
        db_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # Optimizer and loss
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    ce_loss   = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    max_epoch        = max_iterations // len(trainloader) + 1
    iter_num         = 0
    best_performance = 0.0
    early_stop_wait  = 0
    stop_training    = False

    # Training loop
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        if stop_training:
            break

        for sampled_batch in trainloader:

            volume_batch = sampled_batch['image'].to(device)
            label_batch  = sampled_batch['label'].to(device)

            volume_batch = volume_batch.squeeze(1).permute(0, 3, 1, 2).float()

            outputs      = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce   = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss      = 0.5 * (loss_ce + loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            logging.info(
                "iter %d | total_loss=%.4f  ce_loss=%.4f  dice_loss=%.4f" % (
                    iter_num,loss.item(), loss_ce.item(), loss_dice.item(),
                )
            )

            # Validation
            if iter_num % 1 == 0:
                model.eval()
                metric_list = np.zeros((num_classes - 1, 2))

                for val_batch in valloader:
                    metric_i = test_single_image(
                        val_batch["image"], val_batch["label"],
                        model, classes=num_classes,
                        device=device, patch_size=patch_size
                    )
                    metric_list += np.array(metric_i)

                metric_list /= len(db_val)
                mean_dsc  = float(np.mean(metric_list[:, 0]))
                mean_hd95 = float(np.mean(metric_list[:, 1]))

                logging.info(
                    "iter %d | mean_DSC=%.4f  mean_HD95=%.4f" %
                    (iter_num, mean_dsc, mean_hd95)
                )

                if mean_dsc > best_performance:
                    best_performance = mean_dsc
                    early_stop_wait  = 0

                    best_path = os.path.join(snapshot_path, f"{args.model}_best_model.pth")
                    torch.save(model.state_dict(), best_path)

                    ckpt_path = os.path.join(
                        snapshot_path,
                        f"iter_{iter_num}_dsc_{round(best_performance, 4)}.pth"
                    )
                    torch.save(model.state_dict(), ckpt_path)
                    logging.info(f"New best DSC={best_performance:.4f}, saved to {best_path}")

                else:
                    early_stop_wait += 1
                    logging.info(
                        f"No improvement. Early stop counter: "
                        f"{early_stop_wait}/{args.patience}"
                    )
                    if early_stop_wait >= args.patience:
                        logging.info(
                            f"Early stopping at iter {iter_num}. "
                            f"No DSC improvement for {args.patience} consecutive "
                            f"validation checks. Best DSC={best_performance:.4f}."
                        )
                        stop_training = True

                model.train()

            if iter_num >= max_iterations or stop_training:
                break

        if iter_num >= max_iterations:
            break

    logging.info("Training complete.")
    return "Training Finished!"


# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully Supervised Training -- Cataract Surgical Segmentation"
    )

    # Paths
    parser.add_argument('--root_path', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/',
                        help='Root directory of the dataset')

    # Model
    parser.add_argument('--model', type=str, default='swinunet',
                        choices=['unet_resnet', 'swinunet'],
                        help='Model architecture.')
    parser.add_argument('--model_backbone', type=str, default='resnet50',
                        help='Encoder backbone for unet_resnet.')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Output classes including background. '
                             'Paper: 5 (background, iris, pupil, IOL, instrument)')

    # Training schedule
    parser.add_argument('--max_iterations', type=int, default=40000,
                        help='Total training iterations.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size.')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='Initial LR.')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224],
                        help='Input size [H W]. 256 256 (UNet), 224 224 (SwinUNet)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='SGD weight decay.')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='1 = deterministic mode (cudnn.deterministic=True)')

    parser.add_argument('--labeled_num', type=int, default=100,
                        help='Number of labeled training images')

    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (validation checks without improvement)')

    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index')

    # SwinUNet-specific args (only used when --model swinunet)
    swin_group = parser.add_argument_group(
        'SwinUNet',
        'Arguments only used when --model swinunet'
    )
    swin_group.add_argument('--cfg', type=str,
                            default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/code/configs/swin_tiny_patch4_window7_224_lite.yaml',
                            help='Path to Swin config YAML. ')
    swin_group.add_argument('--opts', nargs='+', default=None,
                            help='Override Swin config options as KEY VALUE pairs')
    swin_group.add_argument('--zip', action='store_true',
                            help='Use zipped dataset')
    swin_group.add_argument('--cache-mode', type=str, default='part',
                            choices=['no', 'full', 'part'],
                            help='Dataset caching mode for Swin dataloader')
    swin_group.add_argument('--resume', default=None,
                            help='Resume from checkpoint path')
    swin_group.add_argument('--accumulation-steps', type=int, default=None,
                            help='Gradient accumulation steps')
    swin_group.add_argument('--use-checkpoint', action='store_true',
                            help='Use gradient checkpointing to save memory')
    swin_group.add_argument('--amp-opt-level', type=str, default='O1',
                            choices=['O0', 'O1', 'O2'],
                            help='Mixed precision level (O0 = no AMP)')
    swin_group.add_argument('--tag', default=None,
                            help='Experiment tag')
    swin_group.add_argument('--eval', action='store_true',
                            help='Evaluation only mode')
    swin_group.add_argument('--throughput', action='store_true',
                            help='Throughput test mode')

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.deterministic:
        cudnn.benchmark    = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark    = True
        cudnn.deterministic = False

    train(args)