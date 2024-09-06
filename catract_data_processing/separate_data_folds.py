import pandas as pd
import os
import shutil

# Function to copy images and masks to the respective directories
def copy_files(file_list, img_folder, mask_folder, dest_img_folder, dest_mask_folder):
    for index, row in file_list.iterrows():
        img_name = os.path.basename(row['imgs'])  # Extract image name
        mask_name = os.path.basename(row['masks'])  # Extract mask name

        img_src_path = os.path.join(img_folder, img_name)
        mask_src_path = os.path.join(mask_folder, mask_name)

        # Destination paths for images and masks
        img_dest_path = os.path.join(dest_img_folder, img_name)
        mask_dest_path = os.path.join(dest_mask_folder, mask_name)

        # Check if the image and mask exist and copy them
        if os.path.exists(img_src_path) and os.path.exists(mask_src_path):
            shutil.copy(img_src_path, img_dest_path)
            shutil.copy(mask_src_path, mask_dest_path)
        else:
            print(f"Image or mask not found: {img_name}, {mask_name}")

# Paths to your folders
img_folder = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/all/images'
mask_folder = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/all/masks'

# Paths to destination directories
train_img_dest = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/train/images'
train_mask_dest = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/train/masks'

val_img_dest = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/val/images'
val_mask_dest = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/val/masks'

test_img_dest = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/test/images'
test_mask_dest = '/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/fold_0/test/masks'

# Read the CSV files
train_df = pd.read_csv('/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/Cat1k_anatomy_instrument_train_fold0.csv')
val_df = pd.read_csv('/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/Cat1k_anatomy_instrument_validation_fold0.csv')
test_df = pd.read_csv('/Users/mahtab/Downloads/SSL4MIS/segmentation_env/data/cataract1k/labeled/Cat1k_anatomy_instrument_test_fold0.csv')

# Copy the train files
copy_files(train_df, img_folder, mask_folder, train_img_dest, train_mask_dest)

# Copy the validation files
copy_files(val_df, img_folder, mask_folder, val_img_dest, val_mask_dest)

# Copy the test files
copy_files(test_df, img_folder, mask_folder, test_img_dest, test_mask_dest)

print("Data has been separated into train/val/test folders.")
