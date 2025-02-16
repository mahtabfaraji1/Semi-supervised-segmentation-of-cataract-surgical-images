import os
import zipfile
import shutil
from math import ceil


def split_and_zip_images(folder_path, output_folder, chunk_size=5000):
    """
    Splits images inside folder_path into subfolders, zips them, and removes the subfolders.

    :param folder_path: Path to the folder containing images
    :param output_folder: Path to save the ZIP files
    :param chunk_size: Number of images per ZIP file
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)

    if total_images == 0:
        print("No images found in the folder.")
        return

    num_chunks = ceil(total_images / chunk_size)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_images)
        chunk_files = image_files[start_idx:end_idx]

        subfolder_path = os.path.join(folder_path, f"chunk_{i + 1}")
        os.makedirs(subfolder_path, exist_ok=True)

        # Move images to the subfolder
        for file in chunk_files:
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(subfolder_path, file)
            shutil.move(old_path, new_path)

        # Zip the subfolder
        zip_filename = os.path.join(output_folder, f"images_chunk_{i + 1}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, subfolder_path))  # Store without full path

        print(f"Created: {zip_filename} with {len(chunk_files)} images.")

        # Delete the subfolder to free up space
        shutil.rmtree(subfolder_path)


if __name__ == "__main__":
    input_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/100_labeled/train/images"  # Change this to your images folder path
    output_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/100_labeled/train/images"  # Change this to your desired output folder
    split_and_zip_images(input_folder, output_folder, chunk_size=5000)
