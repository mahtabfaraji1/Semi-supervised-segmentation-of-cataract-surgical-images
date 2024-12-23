import os
import shutil
import random


def create_dirs(base_dir):
    """Create directories for train, val, test splits."""
    os.makedirs(os.path.join(base_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'masks'), exist_ok=True)


def split_data(images_dir, masks_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Get all image files and assume masks have the same names
    images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
    masks = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))])

    assert len(images) == len(masks), "Number of images and masks must be the same."

    # Shuffle the data
    combined = list(zip(images, masks))
    random.shuffle(combined)

    # Calculate split indices
    total_len = len(combined)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    train_data = combined[:train_end]
    val_data = combined[train_end:val_end]
    test_data = combined[val_end:]

    # Create directories
    create_dirs(output_dir)

    def copy_files(data, split):
        for image_file, mask_file in data:
            shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, split, 'images', image_file))
            shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(output_dir, split, 'masks', mask_file))

    # Copy files to corresponding directories
    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    copy_files(test_data, 'test')

    print(f"Data split completed. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


# Example usage
images_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/all/images'  # Path to your images directory
masks_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/all/masks'  # Path to your masks directory
output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/cataract1k/labeled/all'  # Path to the output directory for the split data

split_data(images_dir, masks_dir, output_dir)
