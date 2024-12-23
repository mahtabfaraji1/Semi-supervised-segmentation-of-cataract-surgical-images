import os
import random
import shutil


def select_and_move_images(image_dir, label_dir, target_image_dir, target_label_dir, num_samples=100):
    # Ensure target directories exist
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # List all image files in the directory
    image_files = os.listdir(image_dir)

    # Randomly select the specified number of images
    selected_images = random.sample(image_files, num_samples)

    # Move each selected image and its corresponding label to the target directories
    for image_name in selected_images:
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name)  # Assuming the label file has the same name

        # Destination paths
        target_image_path = os.path.join(target_image_dir, image_name)
        target_label_path = os.path.join(target_label_dir, image_name)

        # Move image and label
        shutil.move(image_path, target_image_path)
        shutil.move(label_path, target_label_path)
        print(f"Moved {image_name} and its label to target folders.")


# Specify the source and destination folders
image_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train/images"
target_image_dir = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train_labeled_500/images"

label_dir ="/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train/labels"
target_label_dir ="/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train_labeled_500/labels"


# Run the function
select_and_move_images(image_dir, label_dir, target_image_dir, target_label_dir, num_samples=100)
