import os
import shutil

def find_and_move_unique_images(folder1, folder2, destination_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get list of image names in both folders
    images_folder1 = set(os.listdir(folder1))
    images_folder2 = set(os.listdir(folder2))

    # Find images that are only in one of the folders (not in both)
    unique_images = (images_folder1.symmetric_difference(images_folder2))

    # Move unique images to the destination folder
    for image_name in unique_images:
        if image_name in images_folder1:
            shutil.move(os.path.join(folder1, image_name), os.path.join(destination_folder, image_name))
        elif image_name in images_folder2:
            shutil.move(os.path.join(folder2, image_name), os.path.join(destination_folder, image_name))

    print(f"Moved {len(unique_images)} unique images to {destination_folder}.")

# Example usage:
folder1 = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1K_labeled_unlabeld/train/images/labeled/'
folder2 = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1k/all/train/images'
destination_folder = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1K_labeled_unlabeld/train/images/unlabeled'

find_and_move_unique_images(folder1, folder2, destination_folder)
