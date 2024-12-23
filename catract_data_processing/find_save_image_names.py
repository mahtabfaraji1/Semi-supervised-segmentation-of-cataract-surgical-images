import os


def save_image_names_to_file(directory, output_file):
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Filter only image files (you can add more extensions if needed)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = [f for f in files if f.lower().endswith(image_extensions)]

    # Write image file names to the output file
    with open(output_file, 'w') as file:
        for image_name in image_files:
            file.write(f"{image_name}\n")
    print(f"Image names saved to {output_file}")


# Usage
directory_path = 'path/to/your/image/directory'  # Replace with your directory path
output_file_path = 'image_names.txt'  # Replace with your desired output file name
save_image_names_to_file(directory_path, output_file_path)
