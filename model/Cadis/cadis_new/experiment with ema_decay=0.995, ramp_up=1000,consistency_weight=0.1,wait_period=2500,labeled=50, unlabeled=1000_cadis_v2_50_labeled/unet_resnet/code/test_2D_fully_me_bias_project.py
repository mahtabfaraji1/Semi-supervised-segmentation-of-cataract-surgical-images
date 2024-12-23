import csv
import argparse
import os
import cv2
import shutil
import numpy as np
import torch
from medpy import metric
from skimage.color import rgb2gray
from skimage.io import imread
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt  # For visualization
from PIL import Image
import numpy as np
from medpy import metric
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/all/test/benchmarking/images/GaussianNoise_5/', help='Directory with input images')
parser.add_argument('--label_dir', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/all/test/labels/', help='Directory with ground truth labels')
parser.add_argument('--exp', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/supervised/unet_all_data_1578/', help='Experiment path')
parser.add_argument('--model', type=str, default='unet', help='Model type')
parser.add_argument('--num_classes', type=int, default=5, help='Number of segmentation classes')
# parser.add_argument('--output_dir', type=str, default='/path/to/output',
#                     help='Output directory for predictions and visualizations')
parser.add_argument('--csv_path', type=str, default='/Users/mahtab/Downloads/DatasetBenchmarking/all_outputs/results.csv', help='CSV file to save metrics')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, asd, hd95


def csv_header_check(csv_path):
    """
    Check if the CSV file already has a header.
    Returns True if headers exist, False otherwise.
    """
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        return headers is not None

def test_single_image(image_path, label_path, net):
    # Load the RGB image and convert it to grayscale
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # image_rgb = imread(image_path)
    # image = imread(image_path)
    # image = rgb2gray(image)  # Convert RGB to grayscale

    # Load the corresponding label
    image = np.array(image)
    label = np.array(label)

    # Resize image and label to patch size (256, 256)
    x, y = image.shape[0], image.shape[1]
    resized_image = zoom(image, (256 / x, 256 / y,1), order=0)
    resized_label = zoom(label, (256 / x, 256 / y), order=0)

    # Prepare the image as input for the network
    input_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().to(device)
    input_tensor = input_tensor.squeeze(1)
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    # Set model to evaluation mode
    net.eval()

    # Run inference on the network
    with torch.no_grad():
        out_main = net(input_tensor)
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        prediction = zoom(out.cpu().detach().numpy(), (x / 256, y / 256), order=0)

    # Save the visualized results
    file_name = os.path.basename(image_path).split('.')[0]
    # visualize_and_save(image, label, prediction, output_dir, file_name)

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

    return metrics_per_class


def Inference(FLAGS):
    image_dir = FLAGS.image_dir
    label_dir = FLAGS.label_dir
    # output_dir = FLAGS.output_dir
    csv_path = FLAGS.csv_path

    metric_dict = {key: [] for key in range(1, FLAGS.num_classes + 1)}

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files = sorted(image_files)

    snapshot_path = "{}{}".format(FLAGS.exp, FLAGS.model)
    net = net_factory(net_type=FLAGS.model, in_chns=3, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path, map_location=device), strict=False)
    net.eval()

    # Check if CSV has headers and open the file in append mode
    write_header = not csv_header_check(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write headers if the file does not have them
        if write_header:
            csv_writer.writerow(
                ["Transformation Name", "Severity Level", "Class", "Mean Dice", "STD Dice", "Mean HD95", "STD HD95"])

        for image_file in tqdm(image_files):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file)

            if not os.path.exists(label_path):
                print(f"Warning: Label for {image_file} not found. Skipping...")
                continue

            # Process the single image and compute metrics
            metrics_per_class = test_single_image(image_path, label_path, net)

            for key, value in metrics_per_class.items():
                metric_dict[key].append(value)

        # Compute the means and standard deviations for each class
        class_names = {1: "Cornea", 2: "Pupil", 3: "Lens", 4: "Instruments"}
        transformation_name = "GaussianNoise"  # Specify the transformation name
        severity_level = "5"  # Specify the severity level

        for key, value in metric_dict.items():
            if key < 5:
                dice, jc, asd, hd95 = zip(
                    *[(value_item[0], value_item[1], value_item[2], value_item[3]) for value_item in value])
                mean_dice, std_dice = round(np.mean(dice), 2), round(np.std(dice), 2)
                mean_hd95, std_hd95 = round(np.mean(hd95), 2), round(np.std(hd95), 2)

                # Write metrics for each class to CSV
                csv_writer.writerow([transformation_name, severity_level, class_names.get(key, 'Unknown'),
                                     mean_dice, std_dice, mean_hd95, std_hd95])
                print(
                    f" {class_names.get(key, 'Unknown')} : dice: mean: {mean_dice}, STD: {std_dice}, hd95: mean: {mean_hd95}, STD: {std_hd95}")

        # Compute and write average metrics across all classes
        total_mean_dice = np.mean(
            [round(np.mean([value_item[0] for value_item in metric_dict[key]]), 2) for key in metric_dict if key < 5])
        total_mean_hd95 = np.mean(
            [round(np.mean([value_item[3] for value_item in metric_dict[key]]), 2) for key in metric_dict if key < 5])
        total_std_dice = np.mean(
            [round(np.std([value_item[0] for value_item in metric_dict[key]]), 2) for key in metric_dict if key < 5])
        total_std_hd95 = np.mean(
            [round(np.std([value_item[3] for value_item in metric_dict[key]]), 2) for key in metric_dict if key < 5])

        total_mean_hd95 = round(total_mean_hd95,2)
        total_std_hd95 = round(total_std_hd95, 2)
        total_mean_dice = round(total_mean_dice, 2)
        total_std_dice = round(total_std_dice, 2)
        # Write overall averages to CSV
        csv_writer.writerow([transformation_name, severity_level, "Overall Average",
                             total_mean_dice, total_std_dice, total_mean_hd95, total_std_hd95])
        print(f"\nOverall Average Dice: Mean = {total_mean_dice}, STD = {total_std_dice}")
        print(f"Overall Average HD95: Mean = {total_mean_hd95}, STD = {total_std_hd95}")


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
