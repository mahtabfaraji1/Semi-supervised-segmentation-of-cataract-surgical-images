import os
import shutil
import cv2
import numpy as np

# Define directories
image_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/images"
mask_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/labels"
output_image_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/images_iris_pupil_class"
output_mask_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/Labels_iris_pupil_class"

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)


# Function to read labels from a mask image
def get_labels(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale
    unique_labels = set(np.unique(mask))  # Extract unique pixel values
    return unique_labels


# Process images and masks
for filename in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, filename)
    image_path = os.path.join(image_dir, filename)

    if os.path.isfile(mask_path) and os.path.isfile(image_path):
        labels = get_labels(mask_path)

        # Check if all labels belong to {0,1,2}
        if labels.issubset({4,0}):
            shutil.copy(image_path, os.path.join(output_image_dir, filename))
            shutil.copy(mask_path, os.path.join(output_mask_dir, filename))
            print(f"Copied: {filename}")

print("Filtering and copying complete.")