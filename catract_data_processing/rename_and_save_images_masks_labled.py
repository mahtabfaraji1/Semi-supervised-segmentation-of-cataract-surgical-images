import os
import shutil


def rename_and_save_images_masks(images_dir, masks_dir, output_images_dir, output_masks_dir):
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Get list of image and mask files (assuming they are in the same order and format)
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))

    # Ensure the same number of images and masks
    assert len(image_files) == len(mask_files), "Mismatch between number of images and masks"

    # Rename and save images and masks
    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files), start=1):
        # Generate new filenames with padding, e.g., "01.png", "02.png", etc.
        new_filename = f"{idx:02d}.png"

        # Define the full paths
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)

        output_image_path = os.path.join(output_images_dir, new_filename)
        output_mask_path = os.path.join(output_masks_dir, new_filename)

        # Copy and rename the image and mask files
        shutil.copy(image_path, output_image_path)
        shutil.copy(mask_path, output_mask_path)

    print(f"Renamed and saved {len(image_files)} images and masks.")


# Example usage
images_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/train/images'
masks_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/train/labels'
output_images_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/renamed/images'
output_masks_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/renamed/labels'

rename_and_save_images_masks(images_directory, masks_directory, output_images_directory, output_masks_directory)
