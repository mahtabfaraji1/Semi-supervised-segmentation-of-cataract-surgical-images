import os
import shutil


def rename_and_save_images_masks(images_dir, output_images_dir,start_index=1749):
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)

    # Get list of image and mask files (assuming they are in the same order and format)
    image_files = sorted(os.listdir(images_dir))


    # Rename and save images and masks
    for idx, (image_file) in enumerate(zip(image_files), start=start_index):
        # Generate new filenames with padding, e.g., "01.png", "02.png", etc.
        if image_file[0] != '.DS_Store':
            new_filename = f"{idx:04d}.png"

            # Define the full paths
            image_path = os.path.join(images_dir, image_file[0])

            output_image_path = os.path.join(output_images_dir, new_filename)

            # Copy and rename the image and mask files
            shutil.copy(image_path, output_image_path)

    print(f"Renamed and saved {len(image_files)} images.")


# Example usage
images_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/unlabeled/raw'
# masks_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/train/labels'
output_images_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/unlabeled/renamed'
# output_masks_directory = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/renamed/labels'

rename_and_save_images_masks(images_directory, output_images_directory,start_index=1749)
