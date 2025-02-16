# import os
import numpy as np
# import cv2
#
# # Configuration: Define your paths
# mask_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/Video01/Labels"  # Replace with the folder containing segmentation masks
# output_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/Video01/Labels_8_class"  # Folder to save the filtered masks
#
# # Mapping of class indices based on your focus
# class_mapping = {
#     # 0: 0,  # Background -> 0
#     0: 0,  # Pupil -> 1
#     1: 1,  # Surgical Tape -> 2
#     2: 2,  # Hand -> 3
#     3: 3,  # Eye Retractors -> 4
#     4: 4,  # Iris -> 5
#     5: 5,  # Skin -> 6
#     6: 6,  # Cornea -> 7
#     # Map all instrument sub-classes to 8 (Instrument)
#     8: 7, 9: 7, 10: 7, 11: 7, 12: 7, 13: 7, 14: 7, 15: 7, 16: 7,
#     17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7, 25: 7, 26: 7,
#     27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7, 34: 7, 35: 7
# }
#
# # Any other class will be set to background (0)
# def process_mask(mask):
#     # Initialize a blank mask with the background value (0)
#     filtered_mask = np.full_like(mask, 0)
#
#     # Map each relevant class
#     for original_class, new_class in class_mapping.items():
#         filtered_mask[mask == original_class] = new_class
#
#     return filtered_mask
#
# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
#
# # Process each mask in the folder
# for filename in os.listdir(mask_folder):
#     if filename.endswith(".png") or filename.endswith(".jpg"):
#         # Load the mask
#         mask_path = os.path.join(mask_folder, filename)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
#         # Process the mask to keep only relevant classes
#         filtered_mask = process_mask(mask)
#
#         # Save the filtered mask
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, filtered_mask)
#
# print("Processing complete! Filtered masks saved in:", output_folder)
import os
import numpy as np
import cv2

# Root folder containing all video folders
main_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test"

# Mapping of class indices based on your focus
# class_mapping = {
#     0: 0,  # Background -> 0
#     1: 1,  # Pupil -> 1
#     2: 2,  # Surgical Tape -> 2
#     3: 3,  # Hand -> 3
#     4: 4,  # Eye Retractors -> 4
#     5: 5,  # Iris -> 5
#     6: 6,  # Skin -> 6
#     7: 7,  # Cornea -> 7
#     # Map all instrument sub-classes to 8 (Instrument)
#     8: 7, 9: 7, 10: 7, 11: 7, 12: 7, 13: 7, 14: 7, 15: 7, 16: 7,
#     17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7, 25: 7, 26: 7,
#     27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7, 34: 7, 35: 7
# }
class_mapping = {
    0: 2,  # Background -> 0
    4: 1,  # Pupil -> 1
    2: 0,  # Surgical Tape -> 2
    3: 0,  # Hand -> 3
    # 4: 4,  # Eye Retractors -> 4
    5: 0,  # Iris -> 5
    6: 0,  # Skin -> 6
    1: 0,
    9: 4, 14: 4, 19: 4,
    # 7: 7,  # Cornea -> 7
    # Map all instrument sub-classes to 8 (Instrument)
    7: 0,8: 0, 10: 0, 11: 0, 12: 0, 13: 0, 15: 0, 16: 0,
    17: 0, 18: 0,  20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0,
    27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0
}
# Function to process a mask
def process_mask(mask):
    # Initialize a blank mask with the background value (0)
    filtered_mask = np.full_like(mask, 0)

    # Map each relevant class
    for original_class, new_class in class_mapping.items():
        filtered_mask[mask == original_class] = new_class

    return filtered_mask

# Loop through each video folder in the main folder
for video_folder in os.listdir(main_folder):
    video_path = os.path.join(main_folder, video_folder)

    # Skip non-directory entries
    if not os.path.isdir(video_path):
        continue

    # Path to the Labels folder in the current video folder
    labels_folder = os.path.join(main_folder, "Labels")
    if not os.path.exists(labels_folder):
        print(f"Labels folder not found in {video_folder}. Skipping...")
        continue

    # Create the Labels_8_class folder inside the video folder
    labels_8_class_folder = os.path.join(main_folder, "Labels_3_class")
    os.makedirs(labels_8_class_folder, exist_ok=True)

    # Process each mask in the Labels folder
    for mask_filename in os.listdir(labels_folder):
        if mask_filename.endswith(".png") or mask_filename.endswith(".jpg"):
            # Load the mask
            mask_path = os.path.join(labels_folder, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Process the mask to keep only relevant classes
            print(f"image_name: {mask_filename} _ labels_four_class: {np.unique(mask)}")
            filtered_mask = process_mask(mask)

            # Save the processed mask in the Labels_8_class folder
            output_path = os.path.join(labels_8_class_folder, mask_filename)
            cv2.imwrite(output_path, filtered_mask)

print("Processing complete! All filtered masks are saved in their respective Labels_3_class folders.")
