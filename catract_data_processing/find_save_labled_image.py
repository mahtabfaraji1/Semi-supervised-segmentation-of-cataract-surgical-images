import os
import shutil


def move_files_from_txt(source_folder, destination_folder, txt_file):
    """
    Reads a txt file containing file names, finds those files in the source folder,
    and copies them to the destination folder.

    Args:
    source_folder (str): Path to the folder where the files are currently stored.
    destination_folder (str): Path to the folder where the files will be copied.
    txt_file (str): Path to the txt file containing the list of file names to move.
    """

    # Ensure the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Read the file names from the txt file
    with open(txt_file, 'r') as file:
        file_names = file.readlines()

    # Clean up the file names (strip any whitespace like \n)
    file_names = [name.strip() for name in file_names]

    # Iterate through the file names and copy them from source to destination
    for file_name in file_names:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        if os.path.exists(source_path):
            # Copy the file to the destination folder
            shutil.copy(source_path, destination_path)
            print(f"Copied: {file_name} to {destination_folder}")
        else:
            print(f"File not found: {file_name}")


# Example usage
source_folder = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1k/all/train/images'
# source_folder = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/all/train/images'
destination_folder = '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1k/90_perc/train/images'
# destination_folder = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/50_perc/train/images'
txt_file =  '/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/cataract1k/90_perc/ratio_txt_files/labeled_images_90_per.txt'
# txt_file = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/50_perc/ratio_txt_files/labeled_images_50_per.txt'

move_files_from_txt(source_folder, destination_folder, txt_file)
