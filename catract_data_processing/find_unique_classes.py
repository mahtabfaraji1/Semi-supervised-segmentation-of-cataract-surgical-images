import os
import numpy as np
from skimage.io import imread

def find_unique_classes(mask_dir):
    unique_classes = set()  # Use a set to store unique class labels_four_class

    # Iterate through all files in the mask directory
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):  # Ensure the file is a .png image
            mask_path = os.path.join(mask_dir, mask_file)
            mask = imread(mask_path)  # Read the mask image

            # Find the unique values (class ,  es) in the mask
            unique_values = np.unique(mask)
            unique_classes.update(unique_values)  # Add them to the set

    return unique_classes

# Directory containing your mask images
mask_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/train/labels_four_class/'

# Find the overall unique classes across all masks
classes = find_unique_classes(mask_directory)
print(f"Overall unique classes: {sorted(classes)}")
