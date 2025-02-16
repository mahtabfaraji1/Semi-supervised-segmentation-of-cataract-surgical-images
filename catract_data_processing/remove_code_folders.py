import os
import shutil

def remove_code_folders(main_folder):
    for root, dirs, files in os.walk(main_folder, topdown=True):
        if "code" in dirs:
            code_path = os.path.join(root, "code")
            print(code_path)
            if os.path.exists(code_path):
                print(f"Removing: {code_path}")
                shutil.rmtree(code_path)
                dirs.remove("code")  # Prevent further traversal into the deleted folder
                print("successful")

if __name__ == "__main__":
    main_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/model/HP"
    if os.path.isdir(main_folder):
        remove_code_folders(main_folder)
    else:
        print("Invalid folder path.")
