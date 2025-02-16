import os
import cv2
import numpy as np
from collections import Counter


def count_classes_in_masks(directory_path):
    """
    Reads all mask images in the specified directory, and counts the occurrence of each class label
    across all images.

    Parameters:
    directory_path (str): Path to the directory containing segmentation mask images.

    Returns:
    Counter: A dictionary with class labels_four_class as keys and their total occurrences as values.
    """
    instance_counts = Counter()

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        # Form the full path to the image
        file_path = os.path.join(directory_path, filename)

        # Check if the file is an image (you can adjust the file extension check as needed)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the mask image in grayscale (assuming each pixel represents a class label)
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Count occurrences of each class in the mask and update the counter
            unique_classes = np.unique(mask)
            for class_label in unique_classes:
                if class_label == 0:
                    # Skip the background class (assuming 0 is background)
                    continue

                # # Create a binary mask for the current class
                # binary_mask = (mask == class_label).astype(np.uint8)
                #
                # # Use connected component analysis to count instances
                # num_labels, _ = cv2.connectedComponents(binary_mask)

                # Subtract 1 from the count to exclude the background label
                instance_counts[class_label] = instance_counts[class_label] +1

    return instance_counts


# Example usage:
directory_path = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/100_labeled_data_semisupervised/test/labels/'
instance_counts = count_classes_in_masks(directory_path)
print("Class counts across all masks:", instance_counts)
