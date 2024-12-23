import os
import shutil


def filter_and_copy_images(all_images_folder, val_folder, test_folder, train_folder):
    # Ensure the train folder exists
    os.makedirs(train_folder, exist_ok=True)

    # Get all image names in each folder
    all_images = {os.path.basename(image) for image in os.listdir(all_images_folder)}
    val_images = {os.path.basename(image) for image in os.listdir(val_folder)}
    test_images = {os.path.basename(image) for image in os.listdir(test_folder)}

    # Find images that are only in the 'all_images' folder and not in 'val' or 'test'
    train_images = all_images - val_images - test_images

    # Copy these images to the train folder
    for image_name in train_images:
        source_path = os.path.join(all_images_folder, image_name)
        destination_path = os.path.join(train_folder, image_name)
        shutil.copy(source_path, destination_path)
        print(f"Copied {image_name} to {train_folder}")


# Specify the paths to each folder
all_images_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/all/all/masks"
val_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/all/val/labels"
test_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/all/test/labels"
train_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/all/train/labels"

# Run the function
filter_and_copy_images(all_images_folder, val_folder, test_folder, train_folder)
