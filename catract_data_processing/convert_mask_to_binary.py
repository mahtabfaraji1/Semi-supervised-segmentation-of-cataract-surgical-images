import os
import numpy as np
from PIL import Image
import cv2


def convert_mask_images_to_binary(directory):
    # Ensure the provided directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if the file is an image
        try:
            img_array = cv2.imread(file_path)
            # with cv2.imread(file_path) as img:
                # Convert image to grayscale if not already
                # img = img.convert("R")

                # Convert pixel values from (0, 255) to (0, 1)
            # img_array = np.array(img)
            if np.max(img_array)> 1:
                print(f"min: {np.min(img_array)} _max: {np.max(img_array)}")
                img_array = (img_array / 255.0).astype(np.uint8)

                # Save the updated image back in the same location
                cv2.imwrite(file_path, img_array)
                # updated_img = Image.fromarray(img_array)
                # updated_img.save(file_path)

                print(f"Converted '{filename}' to binary mask (0, 1).")
            else:
                print(f"image '{filename}' is already a binary mask (0, 1).")
        except Exception as e:
            print(f"Error processing '{filename}': {e}")


# Example usage
directory_path = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/CVS_data/PALM/Validation/Disc_Masks_1"  # Replace with your directory path
convert_mask_images_to_binary(directory_path)
