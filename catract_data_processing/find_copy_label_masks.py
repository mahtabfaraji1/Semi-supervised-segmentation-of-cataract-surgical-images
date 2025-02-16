import os
import shutil

# Define the paths
folder1 = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/1200_labeled/val/labels_four_class"
folder2 = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Cataract1k/all_masks"  # Replace with the path to Folder 2
output_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_40k/1200_labeled/val/labels"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)
# Track files not found
not_found_files = []
# Get the list of files in Folder 1
folder1_files = set(os.listdir(folder1))

# Iterate through Folder 2 to find matching masks
for file_name in os.listdir(folder2):
    if file_name in folder1_files:
        source_file = os.path.join(folder2, file_name)
        if os.path.exists(source_file):
            destination_file = os.path.join(output_folder, file_name)
            shutil.copy(source_file, destination_file)
            print(f"Copied {file_name} to {output_folder}")
        else:
            not_found_files.append(file_name)

        # s

print("Matching masks have been copied successfully!")
# Display masks not found in Folder 2
if not_found_files:
    print("\nThe following masks could not be found in Folder 2:")
    for missing_file in not_found_files:
        print(missing_file)
else:
    print("\nAll masks from Folder 1 were found in Folder 2.")

print("Process completed!")