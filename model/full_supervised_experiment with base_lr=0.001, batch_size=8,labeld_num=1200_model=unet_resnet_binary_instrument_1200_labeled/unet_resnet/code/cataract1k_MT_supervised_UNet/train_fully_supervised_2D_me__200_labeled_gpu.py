import argparse
import logging
import os
import random
import shutil
import sys
import time
import itertools

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_image

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/200_labeled_data_supervised/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Exp_supervised_Unet_200_labeled_unet_resnet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_resnet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=200,
                    help='labeled data')
args = parser.parse_args()



def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
def train(args):
    # base_lr = args.base_lr
    num_classes = args.num_classes
    # batch_size = args.batch_size
    max_iterations = args.max_iterations

    # Define ranges for hyperparameters
    # base_lr_list = [0.0001, 0.001, 0.01, 0.1]
    # batch_size_list = [8, 16, 24, 32]
    base_lr = 0.01
    batch_size = 24
    # for base_lr, batch_size in itertools.product( base_lr_list, batch_size_list):
    exp = f"full_supervised_experiment with base_lr={base_lr}, batch_size={batch_size},labeld_num={args.labeled_num}"
    print(exp)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('..', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))


    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    labeled_idxs = list(range(0, args.labeled_num))
    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
    model.to(device)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    # writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    print("max_epoch:", max_epoch)
    for epoch_num in iterator:
        epoch_num = epoch_num+1
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            volume_batch = volume_batch.squeeze(1)
            volume_batch = volume_batch.permute(0, 3, 1, 2)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_image(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,device=device)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]


                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

                if iter_num >= max_iterations:
                    iter_num =0
                    break
            if iter_num >= max_iterations:
                iter_num = 0
                break
                iterator.close()
    return "Training Finished!"


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

    logging.info(str(args))
    train(args)
