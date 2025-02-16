import os
import shutil
import cv2
import numpy as np

# Define paths
input_images_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/images"  # Update with actual path
input_masks_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/labels"  # Update with actual path
output_images_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/common_images"  # Update with actual path
output_masks_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/common_labels"  # Update with actual path

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Define target labels to check in the masks
target_labels = {9, 14, 19}

# Get list of mask files
mask_files = os.listdir(input_masks_dir)

for mask_file in mask_files:
    mask_path = os.path.join(input_masks_dir, mask_file)

    # Read mask image (assuming grayscale images where pixel values represent labels)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Skipping {mask_file}, unable to read.")
        continue

    # Check if any of the target labels exist in the mask
    unique_values = np.unique(mask)  # Get unique pixel values
    if any(label in unique_values for label in target_labels):
        # Copy mask to output folder
        shutil.copy(mask_path, os.path.join(output_masks_dir, mask_file))

        # Copy corresponding image to output folder
        image_file = mask_file  # Assuming image and mask filenames match
        image_path = os.path.join(input_images_dir, image_file)

        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(output_images_dir, image_file))
            print(f"Copied {image_file} and corresponding mask.")
        else:
            print(f"Warning: No corresponding image found for {mask_file}")

print("Processing complete.")
