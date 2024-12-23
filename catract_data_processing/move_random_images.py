import os
import random
import shutil


def move_random_images(source_folder, target_folder, num_images=1000):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # List all image files in the source folder
    all_images = os.listdir(source_folder)

    # Randomly select the specified number of images
    selected_images = random.sample(all_images, num_images)

    # Move each selected image to the target folder
    for image_name in selected_images:
        source_path = os.path.join(source_folder, image_name)
        target_path = os.path.join(target_folder, image_name)
        shutil.move(source_path, target_path)
        print(f"Moved {image_name} to {target_folder}")


# Define the paths
source_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/unlabeled/images"
target_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/unlabeled/images_1k"

# Run the function
move_random_images(source_folder, target_folder, num_images=1000)
