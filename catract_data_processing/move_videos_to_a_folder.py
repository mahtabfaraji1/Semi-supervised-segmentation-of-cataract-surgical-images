import os
import shutil

# Source directory where folders are located
source_dir = '/home/mahtab/.synapseCache/'  # Change this to your actual source folder
# Destination directory where .mp4 files will be moved
destination_dir = '/home/mahtab/Desktop/catract_videos/'  # Change this to your desired destination folder

# Ensure the destination folder exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Traverse through the directories and subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if the file is an .mp4 file
        if file.endswith('.mp4'):
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(destination_dir, file)

            # Move the file to the destination folder
            shutil.move(source_file_path, destination_file_path)
            print(f"Moved {file} to {destination_dir}")

print("All .mp4 files have been moved.")
