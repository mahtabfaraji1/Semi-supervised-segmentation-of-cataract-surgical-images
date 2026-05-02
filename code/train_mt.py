"""
Reference:
    Faraji et al. "An Evaluation of the Mean Teacher Framework for
    Semi-Supervised Cataract Surgical Image Segmentation."
    Transl Vis Sci Technol. 2026;15(4):5.
    https://doi.org/10.1167/tvst.15.4.5

Usage:
    python train_mt.py \
        --root_path ./data/ \
        --labeled_num 100 \
        --unlabeled_num 40000 \
        --noise_type gaussian_noise \
        --noise_amount 15 \
        --consistency 0.1 \
        --ema_decay 0.995 \
        --ramp_up 1000 \
        --wait_period 2500
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
import torchvision.transforms.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses
from val import test_single_image


# EMA Update
def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Update teacher model weights via exponential moving average (EMA) of
    the student model weights (Eq. 4):

        theta_t = alpha * theta_t + (1 - alpha) * theta_s

    Alpha is warmed up from 0 toward the target value during early iterations
    to avoid the teacher collapsing to a poor initial student.

    Args:
        model:       Student model.
        ema_model:   Teacher model (not directly optimized).
        alpha:       EMA decay rate (paper optimal: 0.995).
        global_step: Current training iteration.
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# Consistency Weight Schedule
def get_current_consistency_weight(rampup_length, current, alpha, wait_period):
    """
    Compute the consistency weight lambda using a sigmoid ramp-up schedule.

    Args:
        rampup_length: Iterations over which lambda ramps up (paper: 1000).
        current:       Current iteration number.
        alpha:         Maximum consistency weight / lambda (paper: 0.1).
        wait_period:   Iterations before ramp-up begins (paper: 2500).

    Returns:
        float: Current value of lambda.
    """
    if current < wait_period:
        return 0.0
    if rampup_length == 0:
        return float(alpha)
    current = np.clip(current - wait_period, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(alpha * np.exp(-5.0 * phase * phase))


# Noise Application

def apply_noise(volume_batch, noise_type, noise_amount):
    """
    Apply input noise to unlabeled images for the student model.

    Noise types evaluated in the paper:
        gaussian_noise: Additive zero-mean Gaussian noise.
                        Severity: sigma = 2 (low), 15 (medium, best), 30 (high).
        gaussian_blur:  Spatial blurring (defocus / motion artifact simulation).
                        Severity: kernel = 3x3, 5x5, 7x7.
        impulse_noise:  Salt-and-pepper noise (high-frequency distortions).
                        Severity: rate = 0.01, 0.03, 0.05 (fraction of pixels).
    Args:
        volume_batch: Unlabeled image tensor.
        noise_type:   One of the types listed above.
        noise_amount: Noise severity parameter (sigma / kernel size / rate).

    Returns:
        Tensor: Noisy input tensor of shape (B, C, H, W), float32.
    """
    if noise_type == "gaussian_noise":
        clamping_value = 2 * noise_amount
        noise = torch.clamp(
            torch.randn_like(volume_batch) * noise_amount,
            -clamping_value, clamping_value
        )
        noisy = torch.clamp(volume_batch + noise, 0, 255)
        noisy = noisy.squeeze(1).permute(0, 3, 1, 2)

    elif noise_type == "gaussian_blur":
        noisy = volume_batch.squeeze(1).permute(0, 3, 1, 2)
        kernel_size = int(noise_amount)
        if kernel_size % 2 == 0:
            kernel_size += 1
        noisy = F.gaussian_blur(noisy, kernel_size=[kernel_size, kernel_size])

    elif noise_type == "impulse_noise":
        noisy = volume_batch.clone()
        prob = noise_amount
        rand = torch.rand_like(noisy)
        noisy[rand < (prob / 2)] = 255                               # salt
        noisy[(rand >= (prob / 2)) & (rand < prob)] = 0              # pepper
        noisy = noisy.squeeze(1).permute(0, 3, 1, 2)
    else:
        noisy = volume_batch.squeeze(1).permute(0, 3, 1, 2)

    return noisy.float()


# Labeled / Unlabeled Index Detection
def find_label_unlabel_index(db_train):
    """
    Identify labeled and unlabeled sample indices by matching image filenames
    against available label filenames.

    The dataset directory holds all images (labeled + unlabeled) together.
    Images whose filenames have a matching label file are labeled; the rest
    are unlabeled. This gives the indices needed by TwoStreamBatchSampler.

    Args:
        db_train: Dataset with .image_paths and .label_paths attributes.

    Returns:
        labeled_indices (list[int]):   Indices of samples that have a label.
        unlabeled_indices (list[int]): Indices of samples without a label.
    """
    image_names = [os.path.basename(p) for p in db_train.image_paths]
    label_names = set(os.path.basename(p) for p in db_train.label_paths)

    labeled_indices   = [i for i, n in enumerate(image_names) if n in label_names]
    unlabeled_indices = [i for i, n in enumerate(image_names) if n not in label_names]

    return labeled_indices, unlabeled_indices


def worker_init_fn(worker_id):
    """Fix random seed per DataLoader worker for reproducibility (seed=1337)."""
    random.seed(1337 + worker_id)


# Training

def train(args):
    # Unpack args
    root_path = args.root_path
    batch_size = args.batch_size
    base_lr = args.base_lr
    ema_decay = args.ema_decay
    consistency = args.consistency
    ramp_up = args.ramp_up
    wait_period = args.wait_period
    model_name = args.model_name
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    unlabeled_num = args.unlabeled_num
    labeled_num = args.labeled_num
    patch_size = args.patch_size
    noise_amount = args.noise_amount
    noise_type = args.noise_type

    labeled_bs = batch_size // 2

    # Device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {list(range(torch.cuda.device_count()))}")

    # Experiment name and output directory
    exp_name = (
        f"labeled={labeled_num}"
        f"_unlabeled={unlabeled_num}"
        f"_ema={ema_decay}"
        f"_lambda={consistency}"
        f"_classes={num_classes}"
        f"_noise={noise_type}"
        f"_amount={noise_amount}"
    )
    snapshot_path = os.path.join("../model", exp_name, model_name)
    os.makedirs(snapshot_path, exist_ok=True)
    print(f"Checkpoints saved to: {snapshot_path}")

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

    # Datasets
    db_train = BaseDataSets(
        base_dir=root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(patch_size)])
    )
    db_val = BaseDataSets(base_dir=root_path, split="val")

    # Identify which samples have labels
    labeled_idxs, unlabeled_idxs = find_label_unlabel_index(db_train)
    print(f"Labeled: {len(labeled_idxs)} | Unlabeled: {len(unlabeled_idxs)}")

    # Save split filenames for reproducibility
    with open(os.path.join(snapshot_path, f"labeled_images_{labeled_num}.txt"), 'w') as f:
        for name in db_train.get_image_names(labeled_idxs):
            f.write(name + "\n")
    with open(os.path.join(snapshot_path, f"unlabeled_images_{unlabeled_num}.txt"), 'w') as f:
        for name in db_train.get_image_names(unlabeled_idxs):
            f.write(name + "\n")


    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        batch_size, batch_size - labeled_bs
    )
    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler,
        num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # Models
    def create_model(ema=False):
        model = net_factory(
            net_type=model_name,
            in_chns=3,
            class_num=num_classes,
            backbone=args.model_backbone
        )
        model.to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(ema=False)
    ema_model = create_model(ema=True)
    model.train()

    # Optimizer and Loss Functions
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info(f"{len(trainloader)} iterations per epoch")
    max_epoch = max_iterations // len(trainloader) + 1
    iter_num = 0
    best_performance = 0.0
    early_stop_wait = 0
    stop_training = False

    # Training Loop
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        if stop_training:
            break
        for sampled_batch in trainloader:

            volume_batch = sampled_batch['image'].to(device)  # (B, 1, H, W, C)
            label_batch = sampled_batch['label'].to(device)  # (B, H, W)

            unlabeled_volume_batch = volume_batch[labeled_bs:]

            # Noise
            ema_inputs = apply_noise(unlabeled_volume_batch, noise_type, noise_amount)

            volume_batch = volume_batch.squeeze(1).permute(0, 3, 1, 2).float()

            # Forward passes
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            # Supervised loss
            loss_ce = ce_loss(
                outputs[:labeled_bs],
                label_batch[:labeled_bs].long()
            )
            loss_dice = dice_loss(
                outputs_soft[:labeled_bs],
                label_batch[:labeled_bs].unsqueeze(1)
            )
            supervised_loss = 0.5 * (loss_ce + loss_dice)

            # Consistency loss
            consistency_loss = torch.mean(
                (outputs_soft[labeled_bs:] - ema_output_soft) ** 2
            )

            # Ramp up lambda
            consistency_weight = get_current_consistency_weight(
                rampup_length=ramp_up,
                current=iter_num,
                alpha=consistency,
                wait_period=wait_period
            )

            # Total loss
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update for teacher
            update_ema_variables(model, ema_model, ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            logging.info(
                "iter %d | total_loss=%.4f  ce_loss=%.4f  dice_loss=%.4f  "
                "cons_loss=%.4f" % (
                    iter_num, loss.item(), loss_ce.item(),
                    loss_dice.item(), consistency_loss.item(),

                )
            )

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
                mean_dsc = float(np.mean(metric_list[:, 0]))
                mean_hd95 = float(np.mean(metric_list[:, 1]))

                logging.info(
                    "iter %d | mean_DSC=%.4f  mean_HD95=%.4f" %
                    (iter_num, mean_dsc, mean_hd95)
                )

                if mean_dsc > best_performance:
                    best_performance = mean_dsc
                    early_stop_wait = 0

                    best_path = os.path.join(snapshot_path, f"{model_name}_best_model.pth")
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
                            f"Early stopping triggered at iter {iter_num}. "
                            f"No DSC improvement for {args.patience} consecutive "
                            f"validation checks (every 200 iterations). "
                            f"Best DSC={best_performance:.4f}."
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
        description="Mean Teacher Semi-Supervised Training -- Cataract Surgical Segmentation"
    )
    parser.add_argument('--root_path', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/',
                        help='Root directory of the dataset')
    parser.add_argument('--model_name', type=str, default='unet_resnet',
                        help='Network type (see networks/net_factory.py). Paper: unet_resnet')
    parser.add_argument('--model_backbone', type=str, default='resnet50',
                        help='Encoder backbone. Paper: resnet50 with ImageNet pretraining')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of output classes including background. '
                             '5 (background, iris, pupil, IOL, instrument)')
    parser.add_argument('--max_iterations', type=int, default=40000,
                        help='Total training iterations.')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Total batch size.')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='Input spatial size [H, W].')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='1 = deterministic (cudnn.deterministic=True)')
    parser.add_argument('--labeled_num', type=int, default=100,
                        help='Number of labeled training images.')
    parser.add_argument('--unlabeled_num', type=int, default=40000,
                        help='Number of unlabeled training images.')
    parser.add_argument('--consistency', type=float, default=0.1,
                        help='Max consistency weight (lambda).')
    parser.add_argument('--ramp_up', type=float, default=1000.0,
                        help='Sigmoid ramp-up length in iterations.')
    parser.add_argument('--wait_period', type=int, default=2500,
                        help='Iterations before ramp-up begins')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='EMA decay rate (alpha) for teacher update.')
    parser.add_argument('--noise_type', type=str, default='gaussian_noise',
                        choices=['gaussian_noise', 'gaussian_blur',
                                 'impulse_noise', 'color_jitter', 'none'],
                        help='Noise applied to student unlabeled input.')
    parser.add_argument('--noise_amount', type=float, default=15,
                        help='Noise severity. Gaussian: sigma (paper best=15); '
                             'Blur: kernel size (3/5/7); Impulse: pixel rate (0.01/0.03/0.05)')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        cudnn.benchmark    = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark    = True
        cudnn.deterministic = False

    train(args)