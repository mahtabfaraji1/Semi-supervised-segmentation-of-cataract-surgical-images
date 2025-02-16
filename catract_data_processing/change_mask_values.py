import os
import cv2
import numpy as np


def invert_mask_images(input_dir, output_dir):
    """Invert black and white colors in BMP mask images and save as PNG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".bmp"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Skipping {filename}, unable to read image.")
                continue

            inverted_img = cv2.bitwise_not(img)

            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, inverted_img)
            print(f"Processed: {filename} -> {output_filename}")


if __name__ == "__main__":
    input_directory = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/CVS_data/PALM/Test/Disc_Masks"  # Change this to your actual input directory
    output_directory = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/CVS_data/PALM/Test/Disc_Masks_1"  # Change this to your actual output directory
    invert_mask_images(input_directory, output_directory)