import json

file_path = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/all_cases/case_5299/ann/case5299_44.png.json'
try:
    with open(file_path, "r") as read_file:
        data = json.load(read_file)
    print("JSON file is valid and well-structured.")
except json.JSONDecodeError as e:
    print(f"Error: {e}")