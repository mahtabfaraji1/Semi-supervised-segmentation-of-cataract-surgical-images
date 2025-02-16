import os
import shutil

# Define paths
images_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/1200_labeled/train/images"  # Folder containing both labeled and unlabeled images
masks_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/1200_labeled/train/labels"    # Folder containing masks (labels) for labeled images
labeled_images_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k/1200_labeled/train/images"  # Destination folder for labeled images

# Ensure destination folder exists
os.makedirs(labeled_images_folder, exist_ok=True)

# Get the list of mask filenames without extensions
mask_filenames = {os.path.splitext(f)[0] for f in os.listdir(masks_folder) if os.path.isfile(os.path.join(masks_folder, f))}

# Iterate through images and copy only labeled ones
for image_file in os.listdir(images_folder):
    image_name, image_ext = os.path.splitext(image_file)

    if image_name in mask_filenames:  # Check if corresponding mask exists
        source_path = os.path.join(images_folder, image_file)
        destination_path = os.path.join(labeled_images_folder, image_file)
        shutil.copy2(source_path, destination_path)  # Copy file with metadata

print("Labeled images copied successfully!")
