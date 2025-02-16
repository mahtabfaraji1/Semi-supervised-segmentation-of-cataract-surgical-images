import os
import random
import shutil

# Set up paths
image_dir = '/home/mahtab/nas/cataract1k/all/train/images'  # Replace with your image directory path
mask_dir = '/home/mahtab/nas/cataract1k/all/train/labels_four_class'  # Replace with your mask directory path
target_image_dir = '/home/mahtab/nas/cataract1k/data_100/images'  # Directory to move selected images
target_mask_dir = '/home/mahtab/nas/cataract1k/data_100/labels_four_class'  # Directory to move selected masks

# Create target directories if they don't exist
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_mask_dir, exist_ok=True)

# List all images and masks
image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

# Ensure the image and mask filenames correspond
image_files = [f for f in image_files if f in mask_files]

# Randomly select 100 images
selected_images = random.sample(image_files, 100)

# Move selected image-mask pairs
for image_file in selected_images:
    mask_file = image_file  # Assuming mask names are the same as image names

    # Define source and destination paths
    src_image_path = os.path.join(image_dir, image_file)
    src_mask_path = os.path.join(mask_dir, mask_file)

    dest_image_path = os.path.join(target_image_dir, image_file)
    dest_mask_path = os.path.join(target_mask_dir, mask_file)

    # Move files
    shutil.move(src_image_path, dest_image_path)
    shutil.move(src_mask_path, dest_mask_path)

print(f"Moved {len(selected_images)} image-mask pairs to target directories.")
