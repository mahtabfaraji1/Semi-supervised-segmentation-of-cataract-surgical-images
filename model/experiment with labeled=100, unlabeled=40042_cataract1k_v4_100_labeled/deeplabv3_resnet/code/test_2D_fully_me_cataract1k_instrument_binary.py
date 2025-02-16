import argparse
import os
import cv2
import shutil
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
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
parser.add_argument('--image_dir', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/100_labeled_data_semisupervised/test/images',
                    help='Name of Experiment')
parser.add_argument('--label_dir', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/100_labeled_data_semisupervised/test/labels',
                    help='experiment_name')
parser.add_argument('--exp', type=str,
                    default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/full_supervised_experiment with base_lr=0.01, batch_size=24,labeld_num=1200_num_claas=2_1200_labeled/',
                    help='experiment_name')
parser.add_argument('--model', type=str, default='unet_resnet', help='model_name')
# parser.add_argument('--model', type=str, default='ViT_Seg', help='model_name')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/full_supervised_experiment with base_lr=0.01, batch_size=24,labeld_num=1200_num_claas=2_1200_labeled/output_cadis/',
                    help='Directory to save output predictions and visualizations')
# parser.add_argument('--patch_size', type=list,  default=[256, 256],
#                     help='patch size of network input')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_colors = {
    1: "Red",
    # 2: "Green",
    # 3: "Blue",
    # 4: "Yellow"
}
class_names = {
    1: "Instrument",
    # 2: "Pupil",
    # 3: "Lens",
    # 4: "Instrument"
}
def compute_dice_score(mask1, mask2):
    """
    Compute IoU, Precision, Recall, F1-score, Pixel Accuracy, and Hausdorff Distance.

    Parameters:
    mask1 (numpy.ndarray): Binary mask of shape (H, W), where 1 indicates the object and 0 is the background.
    mask2 (numpy.ndarray): Binary mask of shape (H, W), where 1 indicates the object and 0 is the background.

    Returns:
    dict: A dictionary containing IoU, Precision, Recall, F1-score, Pixel Accuracy, and Hausdorff Distance.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("The two masks must have the same shape.")

    # True Positives (TP): Pixels correctly classified as 1
    tp = np.logical_and(mask1 == 1, mask2 == 1).sum()

    # True Negatives (TN): Pixels correctly classified as 0
    tn = np.logical_and(mask1 == 0, mask2 == 0).sum()

    # False Positives (FP): Pixels incorrectly classified as 1
    fp = np.logical_and(mask1 == 0, mask2 == 1).sum()

    # False Negatives (FN): Pixels incorrectly classified as 0
    fn = np.logical_and(mask1 == 1, mask2 == 0).sum()

    # Total Pixels: Sum of all classes
    total_pixels = tp + tn + fp + fn

    # IoU: Intersection over Union
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    # dice: Dice score
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)
    pixel_accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0

    # Extract corners from both masks
    # contour_pred = find_largest_contour(mask2, min_contour_area=1000)
    # contour_gt = find_largest_contour(mask1, min_contour_area=1000)
    # corners_pred = extract_corners_from_mask(mask2)
    # corners_gt = extract_corners_from_mask(mask1)

    # Hausdorff Distance
    # hausdorff_dist = hausdorff_distance(np.array(corners_pred), np.array(corners_gt))

    return {
        "IoU": round(iou,2),
        "Precision": round(precision,2),
        "Recall": round(recall,2),
        "F1-Score": round(f1,2),
        "Pixel Accuracy": round(pixel_accuracy,2),
        "Dice":round(dice,2)
        # "Hausdorff Distance": round(hausdorff_dist,2)
    }
def calculate_metric_percase(pred, gt):

    pred = (pred > 0).astype(np.uint8)  # Convert to binary
    gt = (gt > 0).astype(np.uint8)  # Convert to binary

    # If there are any predicted positives, compute Dice and HD95
    if pred.sum() > 0:
        # dice_2=compute_dice_score(pred, gt)
        dice = metric.binary.dc(pred, gt)  # Dice coefficient
        hd95 = metric.binary.hd95(pred, gt)  # 95th percentile Hausdorff distance
        return dice, hd95
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    # dice = metric.binary.dc(pred, gt) * 100.0
    # jc = metric.binary.jc(pred, gt) * 100.0
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)

    # dice = metric.binary.dc(pred, gt)
    # intersection = np.logical_and(pred, gt).sum()
    # union = np.logical_or(pred, gt).sum()
    # iou = intersection / union if union > 0 else 0
    # return dice, jc,asd,hd95

# Define a custom color map (e.g., 5 classes)
def create_custom_colormap():
    # Colors for each class (in RGB format)
    colors = [
        (0, 0, 0),         # Class 0 - Background (black)
        (255, 0, 0),       # Class 1 - Red
        (0, 255, 0),       # Class 2 - Green
        (0, 0, 255),       # Class 3 - Blue
        (255, 255, 0),     # Class 4 - Yellow
    ]
    colors = np.array(colors) / 255.0  # Normalize the RGB values to [0,1]
    cmap = ListedColormap(colors)
    return cmap

def apply_custom_colormap(label, num_classes=5):
    """
    Apply a custom color map to the segmentation mask.
    Args:
        label: The segmentation mask (2D NumPy array with class values).
        num_classes: Number of classes in the segmentation mask.
    Returns:
        RGB image with the applied color map.
    """
    # Create a custom color map for the given number of classes
    cmap = create_custom_colormap()

    # Create an RGB image where the mask is colored according to the custom color map
    label_colored = cmap(label / (num_classes - 1))  # Normalize class values to [0,1] for the color map
    label_colored = (label_colored[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB and scale to [0, 255]

    return label_colored


from PIL import Image, ImageDraw, ImageFont

def visualize_and_save(image,
                       label,
                       prediction,
                       output_path,
                       file_name,
                       num_classes=5,
                       metrics_per_class=None):
    """
    Visualizes and saves the original image, ground truth, and prediction side-by-side.
    Improves readability with a semi-transparent background and dynamic font size.
    """
    # Convert image to np.uint8 if needed
    image = np.asarray(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    # Convert the label and prediction to color
    label_colored = apply_custom_colormap(label, num_classes)
    prediction_colored = apply_custom_colormap(prediction, num_classes)

    # Convert arrays to PIL images
    image_pil = Image.fromarray(image)
    label_pil = Image.fromarray(label_colored)
    prediction_pil = Image.fromarray(prediction_colored)

    # Concatenate the images horizontally
    total_width = image_pil.width + label_pil.width + prediction_pil.width
    max_height = max(image_pil.height, label_pil.height, prediction_pil.height)
    new_img = Image.new('RGB', (total_width, max_height))

    # Paste them side by side
    new_img.paste(image_pil, (0, 0))
    new_img.paste(label_pil, (image_pil.width, 0))
    new_img.paste(prediction_pil, (image_pil.width + label_pil.width, 0))

    # Prepare to draw text
    draw = ImageDraw.Draw(new_img)

    # Use a font size dynamically based on image width
    font_size = max(16, total_width // 80)
    try:
        font = ImageFont.truetype("/Users/mahtab/Downloads/SSL4MIS/segmentation_project/Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text box size
    text_padding = 5
    text_x = 10
    text_y = 10
    line_height = font_size + 5

    # Create a semi-transparent rectangle background for text
    if metrics_per_class:
        num_lines = 1 + len(metrics_per_class)  # One for the image name + one per class
        text_box_height = num_lines * line_height + 2 * text_padding
        text_box_width = total_width // 3
        background = (0, 0, 0, 128)  # Semi-transparent black
        draw.rectangle([(text_x - text_padding, text_y - text_padding),(text_x + text_box_width, text_y + text_box_height)],fill=background)


    # Draw the image name
    draw.text((text_x, text_y), f"Image: {file_name}", fill=(255, 255, 255), font=font)
    text_y += line_height

    # Draw metrics with color-coded class names
    if metrics_per_class:
        for cls_id, (dice, jc, asd, hd95) in metrics_per_class.items():
            color_name = class_colors.get(cls_id, "Unknown")
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
            legend_text = f"{color_name}: {cls_name} (DSC: {dice:.2f})"
            draw.text((text_x, text_y), legend_text, fill=(255, 255, 255), font=font)
            text_y += line_height

    # Save the visualization
    out_path = os.path.join(output_path, f"{file_name}_visualization.png")
    new_img.save(out_path)

# def visualize_and_save(image, label, prediction, output_path, file_name, num_classes=5):
#     """
#     Visualizes and saves the original image (in RGB), ground truth (label), and prediction
#     using a custom color map for the label and prediction.
#     """
#     # Ensure the inputs are NumPy arrays
#     image = np.asarray(image)
#     label = np.asarray(label)
#     prediction = np.asarray(prediction)
#
#     # Ensure all inputs are in the correct format (uint8 for PIL)
#     if image.max() <= 1:  # Assuming the image is normalized between 0 and 1
#         image = (image * 255).astype(np.uint8)
#
#     # Ensure the image is 3D (RGB)
#     if len(image.shape) == 2:
#         raise ValueError("Expected an RGB image, but got a grayscale image. The image should be 3D (H, W, 3).")
#
#     # Convert the original image to PIL format (RGB)
#     image_pil = Image.fromarray(image)
#
#     # Apply the custom color map to the label and prediction
#     label_colored = apply_custom_colormap(label, num_classes)
#     prediction_colored = apply_custom_colormap(prediction, num_classes)
#
#     # Convert the label and prediction to PIL images
#     label_pil = Image.fromarray(label_colored)
#     prediction_pil = Image.fromarray(prediction_colored)
#
#     # Concatenate the images horizontally
#     total_width = image_pil.width + label_pil.width + prediction_pil.width
#     max_height = max(image_pil.height, label_pil.height, prediction_pil.height)
#
#     # Create a blank image with total width and max height
#     new_img = Image.new('RGB', (total_width, max_height))
#
#     # Paste images side by side
#     new_img.paste(image_pil, (0, 0))
#     new_img.paste(label_pil, (image_pil.width, 0))
#     new_img.paste(prediction_pil, (image_pil.width + label_pil.width, 0))
#
#     # Save the new concatenated image
#     new_img.save(os.path.join(output_path, f"{file_name}_visualization.png"))

    # print(f"Visualization saved for {file_name}")
def test_single_image(image_path, label_path, net, output_dir, FLAGS):
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
    patch_size = FLAGS.patch_size[0]
    x, y = image.shape[0], image.shape[1]
    resized_image = zoom(image, (patch_size / x, patch_size / y,1), order=0)
    resized_label = zoom(label, (patch_size / x, patch_size / y), order=0)

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
        # prediction = zoom(out.cpu().detach().numpy(), (x / patch_size, y / patch_size), order=0)
        prediction = out.cpu().detach().numpy()
    # # Save the visualized results
    # file_name = os.path.basename(image_path).split('.')[0]
    # visualize_and_save(image, label, prediction, output_dir, file_name)

    # Find the unique classes in the label
    unique_classes = np.unique(resized_label)

    # Calculate metrics for the detected classes
    metrics_per_class = {}
    for cls in unique_classes:
        if cls == 0:
            continue
        pred_mask = (prediction == cls).astype(np.uint8)
        label_mask = (resized_label == cls).astype(np.uint8)

        # Skip empty classes
        if np.sum(pred_mask) == 0 or np.sum(label_mask) == 0:
            continue
        # Ensure there are non-zero values before calculating metrics
            # Calculate metrics
        metrics_per_class[cls] = calculate_metric_percase(pred_mask, label_mask)

        # if np.sum(pred_mask) > 0 and np.sum(label_mask) > 0:
        #     metrics_per_class[cls] = calculate_metric_percase(pred_mask, label_mask)
        # elif np.sum(pred_mask) == 0 and np.sum(label_mask) > 0:
        #     metrics_per_class[cls] = (0, 0, 0, 0)

    # Visualize and save
    file_name = os.path.basename(image_path).split('.')[0]
    class_names = {1: "Iris", 2: "Pupil", 3: "Lens", 4: "Instruments"}
    # visualize_and_save(image,
    #                    label,
    #                    prediction,
    #                    output_dir,
    #                    file_name,
    #                    FLAGS.num_classes,
    #                    metrics_per_class)


    return metrics_per_class


def Inference(FLAGS):
    image_dir = FLAGS.image_dir
    label_dir = FLAGS.label_dir
    output_dir = FLAGS.output_dir
    metric_dict = {key: [] for key in range(1, FLAGS.num_classes)}

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all PNG files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files = sorted(image_files)
    # image_files = image_files[:213]

    # Initialize the network and load weights
    snapshot_path = "{}{}".format(FLAGS.exp, FLAGS.model)
    net = net_factory(net_type=FLAGS.model, in_chns=3, class_num=FLAGS.num_classes)
    # save_mode_path = os.path.join(snapshot_path, '{}_best_stu_model.pth'.format(FLAGS.model)) #AD_MT
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model)) # mean teacher and supervised UNEt
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(FLAGS.model)) # Co-training CNN&Transformer
    # save_mode_path = os.path.join(snapshot_path, '{}_best_cnn_model.pth'.format(FLAGS.model)) # New Arch_1

    net.load_state_dict(torch.load(save_mode_path, map_location=device),strict=False)  # Load to the appropriate device (CPU/GPU)
    print(f"Initialized model weights from {save_mode_path}")
    net.eval()

    # Iterate over all image files
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file)

        # Ensure that the corresponding label exists
        if not os.path.exists(label_path):
            print(f"Warning: Label for {image_file} not found. Skipping...")
            continue

        # Process the single image and compute metrics
        metrics_per_class = test_single_image(image_path, label_path, net, output_dir, FLAGS)

        for key, value in metrics_per_class.items():
            metric_dict[key].append(value)

    # class_names = {1: "Iris", 2: "Pupil", 3: "Lens", 4: "Instruments"}
    class_names = {1: "Instruments"}
    for key, value in metric_dict.items():
        if key < 2:
            if len(value)>0:
                dice,hd95 = zip(*[(value_item[0], value_item[1]) for value_item in value])
                non_zero_dice = [value for value in dice if value>0.1]
                non_zero_hd95 = [value for value in hd95 if value>0.1]
                print(f" {class_names.get(key, 'Unknown')} : dice: mean:{round(np.mean(non_zero_dice), 2)}, STD: {round(np.std(non_zero_dice), 2)}"
                      f" hd95: mean: {round(np.mean(non_zero_hd95), 2)}, STD: {round(np.std(non_zero_hd95), 2)}")
                # print(f" {class_names.get(key, 'Unknown')} : dice: {round(np.mean(dice),2)}, JC: {round(np.mean(jc),2)}, "
                #       f"asd: {round(np.mean(asd),2)},hd95: {round(np.mean(hd95),2)}")
    with open(os.path.join(output_dir, "metric_dict_sup_300_cataract_40k.pkl"), 'wb') as file:
        pickle.dump(metric_dict, file)
    return metric_dict


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
