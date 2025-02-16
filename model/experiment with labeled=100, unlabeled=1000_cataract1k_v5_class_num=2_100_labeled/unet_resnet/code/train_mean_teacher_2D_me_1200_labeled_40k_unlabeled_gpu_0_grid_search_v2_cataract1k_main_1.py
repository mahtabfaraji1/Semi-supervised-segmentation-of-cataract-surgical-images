import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import itertools
import models
import argparse
import json
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

from dataloaders.dataset_V3_main import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, ramps
from val_2D_main import test_single_volume,test_single_image,test_single_image_cadis

# Setting variables directly
root_path = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/1200_labeled/'
# root_path = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/100_labeled_data_semisupervised/'
# exp = 'Experiment_labeled_50_labeled_1k_unlabeled_gpu_0_resnet_v1'
model_name = 'unet_resnet'
max_iterations = 300000
# batch_size = 24
deterministic = 1
# base_lr = 0.01
patch_size = [256, 256]
seed = 1337
num_classes = 5
labeled_num = 1200
unlabeled_num = 40042
# labeled_bs = round((labeled_num/unlabeled_num) * batch_size)
# labeled_bs = 12
# ema_decay = 0.99
consistency_type = "mse"
# consistency = 0.5
# consistency_rampup = 200.0
# ramp_up, consistency, wait_period = 25, 0.5, 6.0  # Teacher contribution params

# Check for MPS support
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
def get_current_consistency_weight(rampup_length, current, alpha, wait_period=5):
    "rampup_length: The duration over which the consistency weight increases from 0 to its maximum value."
    "current: The current iteration or epoch number."
    "alpha: The maximum value of the consistency weight."
    "wait_period: The initial period during which the consistency weight remains at 0. "
    "This allows the model to learn from labeled data before incorporating the teacher model's consistency loss.)"
    if current < wait_period:
        return 0.0

    else:
        if rampup_length == 0:
            return 1.0
        else:
            current -= wait_period
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(alpha * np.exp(-5.0 * phase * phase))
# def get_current_consistency_weight(epoch):
#     return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

def find_label_unlabel_index(db_train):
    image_paths = db_train.image_paths
    label_paths = db_train.label_paths
    # Extract only the file names (last part of the path)
    image_names = [path.split('/')[-1] for path in image_paths]
    label_names = [path.split('/')[-1] for path in label_paths]

    # Find indices of images that have corresponding labels_four_class
    # Find indices of images that have corresponding labels_four_class
    labeld_indices = [i for i, name in enumerate(image_names) if name in label_names]

    # Find indices of images that do not have corresponding labels_four_class
    unlabeld_indices = [i for i, name in enumerate(image_names) if name not in label_names]

    return labeld_indices, unlabeld_indices


def train():
    # Define ranges for hyperparameters
    # base_lr_list = [0.001, 0.01, 0.1]
    # batch_size_list = [8, 16, 24, 32]
    # ema_decay_list = [0.95, 0.97, 0.99, 0.995, 0.999, 0.9995]
    # ema_decay_list = [0.995, 0.999, 0.9995]
    # ramp_up_list = 1000
    # consistency_list = 0.1
    # wait_period_list = 2500
    ramp_up_list =[1000]
    # wait_period_list = [4000, 5000,6000, 7000, 8000, 9000]

    for ramp_up in ramp_up_list:

        # define hyperparameters
        batch_size = 24
        base_lr = 0.01
        ema_decay = 0.995
        consistency = 0.1
        ramp_up = ramp_up
        wait_period = 2500
        labeled_bs = int((batch_size / 4) / 2)
        # define experiment name
        exp = f"experiment with labeled={labeled_num}, unlabeled={unlabeled_num}_cataract1k_v4"
        print(exp)


        #define global paramenters
        global model_name, num_classes, max_iterations

        # define snapshot paths
        snapshot_path = "../model/{}_{}_labeled/{}".format(
            exp, labeled_num, model_name)
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

        # Clear previous logging handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        # logging.info(f"Experiment Settings: {exp}, Model: {model_name}")



        # Initialize file writers for labeled and unlabeled image names
        labeled_images_file = open(os.path.join(snapshot_path, 'labeled_images_100.txt'), 'w')
        unlabeled_images_file = open(os.path.join(snapshot_path, 'unlabeled_images_1000.txt'), 'w')


        db_train = BaseDataSets(base_dir=root_path, split="train", num=None, transform=transforms.Compose([
            RandomGenerator(patch_size)
        ]))

        db_val = BaseDataSets(base_dir=root_path, split="val")

        # total_images = len(db_train)
        labeled_idxs,unlabeled_idxs = find_label_unlabel_index(db_train)


        # Save the image names in respective text files
        labeled_image_names = db_train.get_image_names(
            labeled_idxs)  # Assuming get_image_names exists or implement a way to fetch image names
        unlabeled_image_names = db_train.get_image_names(unlabeled_idxs)

        for img_name in labeled_image_names:
            labeled_images_file.write(img_name + "\n")

        for img_name in unlabeled_image_names:
            unlabeled_images_file.write(img_name + "\n")

        labeled_images_file.close()
        unlabeled_images_file.close()


        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

        trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                                 num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
        print(f"labeled_idxs: {len(labeled_idxs)}, unlabeld_idxs: {len(unlabeled_idxs)}")

        def create_model(ema=False):
            # Adjust in_chns to 3 for RGB images
            model = net_factory(net_type=model_name, in_chns=3, class_num=num_classes)
            model.to(device)  # Move model to the correct device
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model

        model = create_model()
        ema_model = create_model(ema=True)

        model.train()

        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              momentum=0.9, weight_decay=0.0001)
        ce_loss = CrossEntropyLoss()
        dice_loss = losses.DiceLoss(num_classes)

        # masks_path = os.path.join(root_path,'train','labels_four_class')
        # alpha = losses.calculate_focal_loss_alpha(masks_path)
        # focal_loss = losses.FocalLoss(alpha=alpha)

        # writer = SummaryWriter(snapshot_path + '/log')

        logging.info("{} iterations per epoch".format(len(trainloader)))
        iter_num = 0
        max_epoch = max_iterations // len(trainloader) + 1
        print("max_epoch:",max_epoch)
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(trainloader):
                if iter_num >= max_iterations:
                    # Stop training for this experiment once max_iterations is reached
                    # iter_num = 0
                    break

                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)  # Move data to the correct device
                unlabeled_volume_batch = volume_batch[labeled_bs:]

                noise = torch.clamp(torch.randn_like(
                    unlabeled_volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                ema_inputs = ema_inputs.squeeze(1)
                ema_inputs = ema_inputs.permute(0, 3, 1, 2)

                volume_batch = volume_batch.squeeze(1)
                volume_batch = volume_batch.permute(0, 3, 1, 2)

                outputs = model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output = ema_model(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)
                loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:][:labeled_bs].long())
                # loss_focal = focal_loss(outputs[:labeled_bs], label_batch[:][:labeled_bs].long())

                loss_dice = dice_loss(
                    outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))

                supervised_loss = 0.5 * (loss_dice + loss_ce)

                consistency_loss = torch.mean((outputs_soft[labeled_bs:] - ema_output_soft) ** 2)

                consistency_weight = get_current_consistency_weight(rampup_length=ramp_up, current=iter_num,
                                                                    alpha=consistency, wait_period=wait_period)

                loss = supervised_loss + consistency_weight * consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_ema_variables(model, ema_model, ema_decay, iter_num)
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # print('Learning rate: ', lr_)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                # writer.add_scalar('info/lr', lr_, iter_num)
                # writer.add_scalar('info/total_loss', loss, iter_num)
                # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
                # writer.add_scalar('info/loss_dice', loss_dice, iter_num)
                # writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
                # writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

                logging.info(
                    'iteration %d : total loss : %f, loss_ce: %f, loss_dice: %f, loss_cons: %f , con_weight: %f' %
                    (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_loss.item(),consistency_weight))

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
                    # writer.add_scalar('info/val_mean_dice', performance, iter_num)
                    # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                    if performance > best_performance:
                        best_performance = performance
                        save_mode_path = os.path.join(snapshot_path,
                                                      'iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance, 4)))
                        torch.save(model.state_dict(), save_mode_path)

                        # save student model
                        save_best = os.path.join(snapshot_path,
                                                 '{}_best_model.pth'.format(model_name))

                        torch.save(model.state_dict(), save_best)

                        # save ema model
                        save_best_ema = os.path.join(snapshot_path,
                                                 '{}_best_ema_model.pth'.format(model_name))
                        torch.save(ema_model.state_dict(), save_best_ema)


                    logging.info(
                        'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                    model.train()

                    # break
            # if iter_num >= max_iterations:
                # iter_num = 0
                # iterator.close()
                # break
        # writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/code/config.json')
    args = parser.parse_args()

    # config = json.load(open(args.config))
    train()
