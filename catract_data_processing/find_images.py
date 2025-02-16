import os
import shutil


def copy_unlabeled_images(images_dir, masks_dir, unlabeled_dir):
    # 1) Create the unlabeled directory if it doesn't exist
    if not os.path.exists(unlabeled_dir):
        os.makedirs(unlabeled_dir)

    # 2) Get the base names of all mask files (without extensions)
    #    so we can compare them to image filenames.
    mask_basenames = set()
    for mask_file in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, mask_file)
        if os.path.isfile(mask_path):
            mask_basename, _ = os.path.splitext(mask_file)
            mask_basenames.add(mask_basename)

    # 3) Iterate through images and check if they have corresponding masks
    for img_file in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_file)
        if os.path.isfile(img_path):
            img_basename, _ = os.path.splitext(img_file)

            # If the base name isn't in mask_basenames, it's unlabeled
            if img_basename not in mask_basenames:
                dst_path = os.path.join(unlabeled_dir, img_file)
                shutil.copy2(img_path, dst_path)
                print(f"Copied unlabeled image: {img_file}")
# def copy_matching_images(image_folder, label_folder, target_folder):
#     # Ensure the target folder exists
#     os.makedirs(target_folder, exist_ok=True)
#
#     # Get the list of label filenames without extensions
#     label_names = {os.path.splitext(label)[0] for label in os.listdir(label_folder)}
#
#     # Iterate through all images in the image folder
#     for image in os.listdir(image_folder):
#         # Get the image name without the extension
#         image_name, _ = os.path.splitext(image)
#
#         # If there's a corresponding label, copy the image to the target folder
#         if image_name in label_names:
#             source_path = os.path.join(image_folder, image)
#             target_path = os.path.join(target_folder, image)
#             shutil.copy(source_path, target_path)
#             print(f"Copied {image} to {target_folder}")


# Define the paths
image_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_10k/600_labeled/train/images"
label_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_10k/600_labeled/train/labels_four_class"
target_folder = "/home/mahtab/segmentation/SSL4MIS/segmentation_env/data/Catarct1k_10k/all_unlabaled"

# Run the function
copy_unlabeled_images(image_folder, label_folder, target_folder)
