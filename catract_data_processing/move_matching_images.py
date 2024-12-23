import os
import shutil


def find_and_move_non_matching_images(directory_1, directory_2, directory_3):
    # Ensure directory_3 exists
    os.makedirs(directory_3, exist_ok=True)

    # Get all image names from directory_1
    image_names_in_directory_1 = {os.path.basename(image) for image in os.listdir(directory_1)}

    # Iterate over files in directory_2
    for file_name in os.listdir(directory_2):
        # Check if file is not in the list from directory_1
        if file_name not in image_names_in_directory_1:
            source_path = os.path.join(directory_2, file_name)
            target_path = os.path.join(directory_3, file_name)
            shutil.move(source_path, target_path)
            print(f"Moved {file_name} to {directory_3}")

# Define the paths
source_folder =  "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train_labeled_100/images" # Folder containing the image names
search_folder =  "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train_unlabeled_1000_2/images"# Folder to search for matching images
target_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/labeled/train_unlabeled_1000_3/images"  # Folder to move matching images

# Run the function
find_and_move_non_matching_images(source_folder, search_folder, target_folder)
