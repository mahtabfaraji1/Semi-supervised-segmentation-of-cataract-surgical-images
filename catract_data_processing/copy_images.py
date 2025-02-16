import os
import shutil


def copy_images(src_dir, dst_dir):
    """
    Copies all files from src_dir to dst_dir.
    If dst_dir doesn't exist, it is created.
    """
    # 1) Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 2) Copy files from src_dir to dst_dir
    for filename in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, filename)
        dst_file_path = os.path.join(dst_dir, filename)

        # Only copy if it's a file (not a subdirectory)
        if os.path.isfile(src_file_path):
            shutil.copy2(src_file_path, dst_file_path)
            print(f"Copied {filename} to {dst_dir}")


if __name__ == "__main__":
    source_directory = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/all_unlabaled"
    destination_directory = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/1200_labeled/train/images"

    copy_images(source_directory, destination_directory)
