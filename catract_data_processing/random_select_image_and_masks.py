import os
import random
import shutil

def select_and_copy_images_from_each_case(source_dir, img_folder, mask_folder, dest_img_dir, dest_mask_dir, num_samples=50):
    # Iterate through the cases in the source directory
    case_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for case_dir in case_dirs:
        print(case_dir)
        img_path = os.path.join(source_dir, case_dir, img_folder)
        mask_path = os.path.join(source_dir, case_dir, mask_folder)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            img_files = [f for f in os.listdir(img_path) if
                         os.path.isfile(os.path.join(img_path, f)) and f != ".DS_Store"]
            mask_files = [f for f in os.listdir(mask_path) if
                          os.path.isfile(os.path.join(mask_path, f)) and f != ".DS_Store"]

            # Assuming images and masks have the same filenames
            common_files = [img for img in img_files if img in mask_files]
            # selected_files = random.sample(common_files, min(num_samples, len(common_files)))
            selected_files = random.sample(common_files, num_samples)
            # Create destination directories for each case if they don't exist
            case_dest_img_dir = os.path.join(dest_img_dir, case_dir)
            case_dest_mask_dir = os.path.join(dest_mask_dir, case_dir)
            os.makedirs(case_dest_img_dir, exist_ok=True)
            os.makedirs(case_dest_mask_dir, exist_ok=True)

            # Copy the selected images and masks to the destination directories
            for file_name in selected_files:
                shutil.copy(os.path.join(img_path, file_name), case_dest_img_dir)
                shutil.copy(os.path.join(mask_path, file_name), case_dest_mask_dir)


# Example usage
# Example usage
source_directory = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/labeled/videos"
image_folder_name = "Images"
mask_folder_name = "Labels_8_class"
destination_image_directory = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/labeled/video_50_random/img"
destination_mask_directory = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/labeled/video_50_random/masks"

select_and_copy_images_from_each_case(source_directory, image_folder_name, mask_folder_name, destination_image_directory, destination_mask_directory)

# import os
# import random
# import shutil
#
# def select_and_copy_images_from_each_case(source_dir, img_folder, dest_img_dir, num_samples=50):
#     # Iterate through the cases in the source directory
#     case_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
#
#     for case_dir in case_dirs:
#         print(case_dir)
#         img_path = os.path.join(source_dir, case_dir, img_folder)
#
#         if os.path.exists(img_path):
#             img_files = [f for f in os.listdir(img_path) if
#                          os.path.isfile(os.path.join(img_path, f)) and f != ".DS_Store"]
#
#             # Assuming images and masks have the same filenames
#             # common_files = [img for img in img_files if img in mask_files]
#             # selected_files = random.sample(common_files, min(num_samples, len(common_files)))
#             selected_files = random.sample(img_files, num_samples)
#             # Create destination directories for each case if they don't exist
#             case_dest_img_dir = os.path.join(dest_img_dir, case_dir)
#             # case_dest_mask_dir = os.path.join(dest_mask_dir, case_dir)
#             os.makedirs(case_dest_img_dir, exist_ok=True)
#             # os.makedirs(case_dest_mask_dir, exist_ok=True)
#
#             # Copy the selected images and masks to the destination directories
#             for file_name in selected_files:
#                 shutil.copy(os.path.join(img_path, file_name), case_dest_img_dir)
#                 # shutil.copy(os.path.join(mask_path, file_name), case_dest_mask_dir)
#
#
# # Example usage
# source_directory = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/labeled/videos"
# image_folder_name = "Images"
# mask_folder_name = "Labels_8_class"
# destination_image_directory = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/labeled/video_50_random/img"
# destination_mask_directory = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/labeled/video_50_random/masks"
#
# select_and_copy_images_from_each_case(source_directory, image_folder_name, destination_image_directory)
