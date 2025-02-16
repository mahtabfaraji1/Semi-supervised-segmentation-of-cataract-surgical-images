# import os
# import shutil
# import numpy as np
# from PIL import Image
#
# # Input and output paths
# base_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/videos"  # Replace with the path to the folder containing subfolders like Video03
# output_images_dir = os.path.join(base_dir, "output_images")
# output_labels_dir = os.path.join(base_dir, "Labels")
#
# # Ensure output directories exist
# os.makedirs(output_images_dir, exist_ok=True)
# os.makedirs(output_labels_dir, exist_ok=True)
#
# # Classes of interest
# target_classes = {9, 14, 19}
#
#
# def check_and_copy_labels_and_images(video_dir):
#     labels_dir = os.path.join(video_dir, "Labels")
#     images_dir = os.path.join(video_dir, "Images")
#
#     if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
#         return
#
#     for label_file in os.listdir(labels_dir):
#         label_path = os.path.join(labels_dir, label_file)
#
#         # Check if the file is a valid image
#         try:
#             label_image = np.array(Image.open(label_path))
#         except Exception as e:
#             print(f"Error reading {label_file}: {e}")
#             continue
#
#         # Check if any target class exists in the label
#         if any(class_id in label_image for class_id in target_classes):
#             # Copy label
#             shutil.copy(label_path, output_labels_dir)
#
#             # Copy corresponding image
#             corresponding_image_path = os.path.join(images_dir, label_file)
#             if os.path.exists(corresponding_image_path):
#                 shutil.copy(corresponding_image_path, output_images_dir)
#
#
# # Loop through each video folder
# for folder_name in os.listdir(base_dir):
#     video_folder = os.path.join(base_dir, folder_name)
#     if os.path.isdir(video_folder):
#         check_and_copy_labels_and_images(video_folder)
#
# print("Processing complete!")
import os
import shutil
import numpy as np
from PIL import Image
import random

# Input and output paths
base_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/videos"
output_images_dir = os.path.join(base_dir, "output_images")
output_labels_dir = os.path.join(base_dir, "Labels")

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Classes of interest
target_classes = {9, 14, 19}

# Containers to keep track of processed files
images_with_target_classes = []
images_without_target_classes = []


def check_and_classify_labels_and_images(video_dir):
    labels_dir = os.path.join(video_dir, "Labels")
    images_dir = os.path.join(video_dir, "Images")

    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        return

    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)

        # Check if the file is a valid image
        try:
            label_image = np.array(Image.open(label_path))
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue

        # Check if any target class exists in the label
        if any(class_id in label_image for class_id in target_classes):
            images_with_target_classes.append((label_path, os.path.join(images_dir, label_file)))
        else:
            images_without_target_classes.append((label_path, os.path.join(images_dir, label_file)))


# Loop through each video folder
for folder_name in os.listdir(base_dir):
    video_folder = os.path.join(base_dir, folder_name)
    if os.path.isdir(video_folder):
        check_and_classify_labels_and_images(video_folder)

# Ensure we have enough images for the requirements
if len(images_with_target_classes) < 169 or len(images_without_target_classes) < 44:
    raise ValueError(
        "Not enough images with or without the target classes to meet the required total of 213 images."
    )

# Randomly select 169 images with target classes and 44 without
selected_with_classes = random.sample(images_with_target_classes, 169)
selected_without_classes = random.sample(images_without_target_classes, 44)

# Copy the selected files to the output folders
for label_path, image_path in selected_with_classes + selected_without_classes:
    # Copy label
    shutil.copy(label_path, output_labels_dir)

    # Copy corresponding image
    if os.path.exists(image_path):
        shutil.copy(image_path, output_images_dir)

print("Processing complete! The output folders contain 213 images, with 169 containing target classes.")
