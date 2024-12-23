import os

# Define the directory where the files are located
directory = '/Users/mahtab/Downloads/Code_RWD_CL/Gluaocma_detection_bias/dataset/not_folder/benchmarking/fundus/1/images/GaussianNoise_1/'

# Define the part of the filename you want to remove
part_to_remove = '_GaussianNoise_1_image'

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Check if the part to remove is in the filename
    if part_to_remove in filename:
        # Create the new filename by replacing the part_to_remove with an empty string
        new_filename = filename.replace(part_to_remove, '')

        # Construct full file paths for renaming
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_file, new_file)

        # Print a message indicating the renaming
        print(f'Renamed: {filename} -> {new_filename}')
