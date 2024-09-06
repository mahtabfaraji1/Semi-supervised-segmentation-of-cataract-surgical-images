import argparse
import os
import shutil
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from skimage.color import rgb2gray  # For RGB to Grayscale conversion
from skimage.io import imread  # For reading images
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/test/images',
                    help='Name of Experiment')
parser.add_argument('--label_dir', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/test/masks',
                    help='experiment_name')

parser.add_argument('--exp', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_env/model/Experiment_labeled_900_900_labeled/',
                    help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=15,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=409,
                    help='labeled data')
parser.add_argument('--output_dir', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_env/model/Experiment_labeled_900_900_labeled/output',
                    help='Directory to save output predictions')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    return dice


def test_single_image(image_path, label_path, net, output_dir, FLAGS):
    # Load the RGB image and convert it to grayscale
    image = imread(image_path)
    image = rgb2gray(image)  # Convert RGB to grayscale

    # Load the corresponding label
    label = imread(label_path)

    # Resize image and label to patch size (256, 256)
    x, y = image.shape[0], image.shape[1]
    resized_image = zoom(image, (256 / x, 256 / y), order=0)
    resized_label = zoom(label, (256 / x, 256 / y), order=0)

    # Prepare the image as input for the network
    input_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().to(device)

    # input_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

    # Set model to evaluation mode
    net.eval()

    # Run inference on the network
    with torch.no_grad():
        out_main = net(input_tensor)
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        prediction = zoom(out.cpu().detach().numpy(), (x / 256, y / 256), order=0)

    # Find the unique classes in the label
    unique_classes = np.unique(label)

    # Calculate metrics for the detected classes
    metrics_per_class = {}
    for cls in unique_classes:
        if cls == 0:  # Skip the background class
            continue
        pred_mask = (prediction == cls).astype(np.uint8)
        label_mask = (label == cls).astype(np.uint8)

        # Ensure there are non-zero values before calculating metrics
        if np.sum(pred_mask) > 0 and np.sum(label_mask) > 0:
            metrics_per_class[cls] = calculate_metric_percase(pred_mask, label_mask)

    # Save the prediction and images
    # case_name = os.path.basename(image_path).split('.')[0]
    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

    # sitk.WriteImage(prd_itk, os.path.join(output_dir, f"{case_name}_pred.nii.gz"))
    # sitk.WriteImage(img_itk, os.path.join(output_dir, f"{case_name}_img.nii.gz"))
    # sitk.WriteImage(lab_itk, os.path.join(output_dir, f"{case_name}_gt.nii.gz"))

    return metrics_per_class


def Inference(FLAGS):
    image_dir = FLAGS.image_dir
    label_dir = FLAGS.label_dir
    output_dir = FLAGS.output_dir
    metric_dict = {key: [] for key in range(1, FLAGS.num_classes + 1)}
    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # List all PNG files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files = sorted(image_files)

    # Initialize the network and load weights
    snapshot_path = "{}{}".format(FLAGS.exp, FLAGS.model)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path, map_location=device))  # Load to the appropriate device (CPU/GPU)
    print(f"Initialized model weights from {save_mode_path}")
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    # Iterate over all image files
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        # label_file = image_file.replace(".png", "_label.png")
        label_path = os.path.join(label_dir, image_file)

        # Ensure that the corresponding label exists
        if not os.path.exists(label_path):
            print(f"Warning: Label for {image_file} not found. Skipping...")
            continue

        # Process the single image and compute metrics
        metrics_per_class = test_single_image(image_path, label_path, net, output_dir, FLAGS)

        for key, value in metrics_per_class.items():
            metric_dict[key].append(value)
            # print(key, value)

        # first_total += np.asarray(first_metric)
        # second_total += np.asarray(second_metric)
        # third_total += np.asarray(third_metric)

    class_names = {1: "Cornea", 2: "Pupil", 3: "Lens", 4: "instruments"}
    for key, value in metric_dict.items():
        if key < 5:
            print(f" Dice_score of {class_names[key]} : {np.mean(value)}")
    return metric_dict


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
    # print(metrics)
    # print((metrics[0] + metrics[1] + metrics[2]) / 3)
