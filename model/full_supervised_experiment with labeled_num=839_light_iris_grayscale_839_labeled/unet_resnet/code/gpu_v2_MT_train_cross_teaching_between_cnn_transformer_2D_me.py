import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
from config import get_config
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, ramps
from val_2D import test_single_image

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Basic experiment settings

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default="1,2", help="comma-separated GPU IDs to use")

parser.add_argument('--root_path', type=str,
                    default='/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1k/all/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Experiment_MT_cnn_transformer_labeled_all', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--transformer_model_name', type=str,
                    default='transformer', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="/home/mahtab/segmentation/SSL4MIS/segmentation_env/code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=789,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


# Convert GPU IDs string into a list of integers
gpu_ids = list(map(int, args.gpu_ids.split(',')))

if torch.cuda.is_available() and len(gpu_ids) > 0:
    device = torch.device(f"cuda:{gpu_ids[0]}")
else:
    device = torch.device("cpu")
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(model_type='cnn', ema=False):
        if model_type == 'cnn':
            model = net_factory(class_num=num_classes, net_type=args.model, in_chns=3, device=device)
        else:
            model = ViT_seg(config, img_size=args.patch_size,
                            num_classes=args.num_classes).to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_images = len(db_train)
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, total_images))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # Create student and teacher models for CNN and Transformer
    cnn_student = create_model(model_type='cnn')
    cnn_teacher = create_model(model_type='cnn', ema=True)

    transformer_student = create_model(model_type='transformer')
    transformer_teacher = create_model(model_type='transformer', ema=True)

    # Multi-GPU wrapping with specific GPU IDs
    if torch.cuda.device_count() > 1 and len(gpu_ids) > 1:
        cnn_student = torch.nn.DataParallel(cnn_student, device_ids=gpu_ids)
        cnn_teacher = torch.nn.DataParallel(cnn_teacher, device_ids=gpu_ids)
        transformer_student = torch.nn.DataParallel(transformer_student, device_ids=gpu_ids)
        transformer_teacher = torch.nn.DataParallel(transformer_teacher, device_ids=gpu_ids)

    # Optimizer for both models
    optimizer1 = optim.SGD(cnn_student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(transformer_student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance_cnn = 0.0
    best_performance_transformer = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            # CNN student and teacher predictions
            volume_batch = volume_batch.squeeze(1)
            volume_batch = volume_batch.permute(0, 3, 1, 2)

            ema_inputs = ema_inputs.squeeze(1)
            ema_inputs = ema_inputs.permute(0, 3, 1, 2)

            outputs_cnn_student = cnn_student(volume_batch)
            outputs_cnn_student_soft = torch.softmax(outputs_cnn_student, dim=1)

            with torch.no_grad():
                outputs_cnn_teacher = cnn_teacher(ema_inputs)
                outputs_cnn_teacher_soft = torch.softmax(outputs_cnn_teacher, dim=1)

            # Transformer student and teacher predictions
            outputs_transformer_student = transformer_student(volume_batch)
            outputs_transformer_student_soft = torch.softmax(outputs_transformer_student, dim=1)

            with torch.no_grad():
                outputs_transformer_teacher = transformer_teacher(ema_inputs)
                outputs_transformer_teacher_soft = torch.softmax(outputs_transformer_teacher, dim=1)

            # Supervised losses (cross-entropy + dice) for both CNN and Transformer
            loss_ce_cnn = ce_loss(outputs_cnn_student[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_dice_cnn = dice_loss(outputs_cnn_student_soft[:args.labeled_bs],
                                      label_batch[:args.labeled_bs].unsqueeze(1))
            cnn_supervised_loss = 0.5 * (loss_dice_cnn + loss_ce_cnn)

            loss_ce_transformer = ce_loss(outputs_transformer_student[:args.labeled_bs],
                                          label_batch[:args.labeled_bs].long())
            loss_dice_transformer = dice_loss(outputs_transformer_student_soft[:args.labeled_bs],
                                              label_batch[:args.labeled_bs].unsqueeze(1))
            transformer_supervised_loss = 0.5 * (loss_dice_transformer + loss_ce_transformer)

            # Consistency loss for CNN and Transformer (on unlabeled data)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            cnn_consistency_loss = torch.mean(
                (outputs_cnn_student_soft[args.labeled_bs:] - outputs_cnn_teacher_soft) ** 2)
            transformer_consistency_loss = torch.mean(
                (outputs_transformer_student_soft[args.labeled_bs:] - outputs_transformer_teacher_soft) ** 2)

            # Cross-teaching loss between CNN and Transformer (on unlabeled data)
            cnn_cross_loss = torch.mean(
                (outputs_cnn_student_soft[args.labeled_bs:] - outputs_transformer_teacher_soft) ** 2)
            transformer_cross_loss = torch.mean(
                (outputs_transformer_student_soft[args.labeled_bs:] - outputs_cnn_teacher_soft) ** 2)

            # Total losses for CNN and Transformer (including cross-teaching)
            cnn_loss = cnn_supervised_loss + consistency_weight * (cnn_consistency_loss + cnn_cross_loss)
            transformer_loss = transformer_supervised_loss + consistency_weight * (
                    transformer_consistency_loss + transformer_cross_loss)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            cnn_loss.backward()
            transformer_loss.backward()

            optimizer1.step()
            optimizer2.step()

            # EMA update for both CNN and Transformer teachers
            update_ema_variables(cnn_student, cnn_teacher, args.ema_decay, iter_num)
            update_ema_variables(transformer_student, transformer_teacher, args.ema_decay, iter_num)

            iter_num += 1

            # Logging
            writer.add_scalar('loss/cnn_loss', cnn_loss, iter_num)
            writer.add_scalar('loss/transformer_loss', transformer_loss, iter_num)
            writer.add_scalar('loss/cnn_cross_loss', cnn_cross_loss, iter_num)
            writer.add_scalar('loss/transformer_cross_loss', transformer_cross_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            logging.info('iteration %d : CNN loss : %f Transformer loss : %f' % (
                iter_num, cnn_loss.item(), transformer_loss.item()))
            # Validation every 200 iterations
            if iter_num % 200 == 0:
                cnn_student.eval()
                transformer_student.eval()
                metric_list_cnn = 0.0
                metric_list_transformer = 0.0
                i = 1
                for i_batch, sampled_batch in enumerate(valloader):

                    metric_i_cnn = test_single_image(sampled_batch["image"], sampled_batch["label"], cnn_student,
                                                     classes=num_classes, device=device,patch_size = args.patch_size)
                    metric_list_cnn += np.array(metric_i_cnn)

                    metric_i_transformer = test_single_image(sampled_batch["image"], sampled_batch["label"],
                                                             transformer_student, classes=num_classes,
                                                             device=device,patch_size = args.patch_size)
                    metric_list_transformer += np.array(metric_i_transformer)


                metric_list_cnn = metric_list_cnn / len(db_val)
                metric_list_transformer = metric_list_transformer / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('val/cnn_val_{}_dice'.format(class_i + 1), metric_list_cnn[class_i, 0], iter_num)
                    writer.add_scalar('val/transformer_val_{}_dice'.format(class_i + 1),
                                      metric_list_transformer[class_i, 0], iter_num)

                performance_cnn = np.mean(metric_list_cnn, axis=0)[0]
                performance_transformer = np.mean(metric_list_transformer, axis=0)[0]

                mean_hd95_cnn = np.mean(metric_list_cnn, axis=0)[1]
                mean_hd95_transformer = np.mean(metric_list_transformer, axis=0)[1]

                writer.add_scalar('val/cnn_mean_dice', performance_cnn, iter_num)
                writer.add_scalar('val/cnn_mean_hd95', mean_hd95_cnn, iter_num)

                writer.add_scalar('val/transformer_mean_dice', performance_transformer, iter_num)
                writer.add_scalar('val/transformer_mean_hd95', mean_hd95_transformer, iter_num)

                # Save best CNN model
                if performance_cnn > best_performance_cnn:
                    best_performance_cnn = performance_cnn
                    save_best_cnn = os.path.join(snapshot_path, '{}_best_cnn_model.pth'.format(args.model))
                    torch.save(cnn_student.state_dict(), save_best_cnn)

                # Save best Transformer model
                if performance_transformer > best_performance_transformer:
                    best_performance_transformer = performance_transformer
                    save_best_transformer = os.path.join(snapshot_path,
                                                         '{}_best_transformer_model.pth'.format(args.transformer_model_name))
                    torch.save(transformer_student.state_dict(), save_best_transformer)

                cnn_student.train()
                transformer_student.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
