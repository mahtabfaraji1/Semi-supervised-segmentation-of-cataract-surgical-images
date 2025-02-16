import os
from PIL import Image

def convert_images_to_png(directory):
    # Ensure the provided directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if the file is an image
        try:
            with Image.open(file_path) as img:
                # Check if the extension is not .png
                if not filename.lower().endswith(".png"):
                    # Create a new filename with .png extension
                    # new_filename = os.path.splitext(filename)[0].split('.')[0] + ".png"
                    new_filename = os.path.splitext(filename)[0] + ".png"
                    new_file_path = os.path.join(directory, new_filename)

                    # Convert and save the image in .png format
                    img.save(new_file_path, "PNG")
                    os.remove(file_path)  # Remove the original file
                    print(f"Converted '{filename}' to '{new_filename}'.")
                else:
                    print(f"'{filename}' is already a .png file.")
        except Exception as e:
            print(f"Error processing '{filename}': {e}")


# Example usage
directory_path = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/CVS_data/OD_all_data/val/images"
convert_images_to_png(directory_path)
