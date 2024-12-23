import os
import shutil


# Ensure the destination directory exists, if not, create it
import os
import shutil

def copy_images(source_folder, target_folder):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # List all files in the source folder
    for image_name in os.listdir(source_folder):
        source_path = os.path.join(source_folder, image_name)
        target_path = os.path.join(target_folder, image_name)

        # Copy only if it is a file (to avoid copying directories)
        if os.path.isfile(source_path):
            shutil.copy(source_path, target_path)
            print(f"Copied {image_name} to {target_folder}")


# Define the paths
# Source directory where case folders are located
source_folder = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/unlabeled/1k/'  # Replace with your actual source directory path

# Destination directory where all images will be moved
target_folder = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1k/train/images/'  # Replace with your desired destination directory


# Run the function
copy_images(source_folder, target_folder)
