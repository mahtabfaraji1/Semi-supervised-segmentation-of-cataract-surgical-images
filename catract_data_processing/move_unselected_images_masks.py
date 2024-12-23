import os
import shutil

def find_and_copy_unselected_images(original_folder, selected_folder, output_folder):
    # Define subfolders for images and masks
    output_images_folder = os.path.join(output_folder, "images")
    output_masks_folder = os.path.join(output_folder, "masks")

    # Create output directories if they don't exist
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)

    # Get the list of selected images (without extensions)
    selected_images = set()
    for case in os.listdir(selected_folder):
        if case !='.DS_Store':
            case_path = os.path.join(selected_folder, case)
            if os.path.isdir(case_path):
                for image_file in os.listdir(os.path.join(case_path)):
                    selected_images.add(os.path.splitext(image_file)[0])

    # Iterate through each case in the original folder
    for case in os.listdir(original_folder):
        case_path = os.path.join(original_folder, case)

        # Skip if not a directory
        if not os.path.isdir(case_path):
            continue

        # Get paths for images and masks
        images_path = os.path.join(case_path, "img")
        masks_path = os.path.join(case_path, "mask_anatomy_inst")

        # Check existence of both images and masks folders
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            continue

        # Iterate over images in the images folder
        for image_file in os.listdir(images_path):
            image_name, ext = os.path.splitext(image_file)
            mask_file = image_name + ext

            # Check if the image is not in the selected folder
            if image_name not in selected_images:
                # Copy image and corresponding mask
                src_image_path = os.path.join(images_path, image_file)
                src_mask_path = os.path.join(masks_path, mask_file)

                dst_image_path = os.path.join(output_images_folder, image_file)
                dst_mask_path = os.path.join(output_masks_folder, mask_file)

                # Copy files
                if os.path.exists(src_image_path):
                    shutil.copy(src_image_path, dst_image_path)
                if os.path.exists(src_mask_path):
                    shutil.copy(src_mask_path, dst_mask_path)



# Example usage
original_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/Catarac1k/labeled_data_cases/train_cases_labeled"  # Path to the main folder containing subfolders for cases
selected_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/Catarac1k/labeled_data_cases/train_images_labeled_random_50/img"  # Path to the folder containing 50 selected images
output_folder = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/Catarac1k/unlabeled_data_cases/from_same_labeled"      # Path to the output folder for unselected images and masks

find_and_copy_unselected_images(original_folder, selected_folder, output_folder)
