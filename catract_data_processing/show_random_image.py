import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Directories containing images and masks
image_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/bias_data_iris_color/light_iris_color/test/images/'
mask_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/bias_data_iris_color/light_iris_color/test/labels/'

# Get list of image and mask filenames
image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

# Choose a random image and its corresponding mask
random_index = random.randint(0, len(image_filenames) - 1)
image_path = os.path.join(image_dir, image_filenames[random_index])
mask_path = os.path.join(mask_dir, mask_filenames[random_index])

# Load the image and mask
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB

# Load the mask and check unique values
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
unique_values = np.unique(mask)
print("Unique values in the mask:", unique_values)

# Class mapping for visualization
class_names = [f"Class {unique_values[idx]}" for idx in range(len(unique_values))]
class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))  # Use a colormap
cmap = ListedColormap(class_colors[:len(unique_values)])

# Map mask values to class indices
class_mapping = {value: idx for idx, value in enumerate(unique_values)}
mask_classes = np.zeros_like(mask)
for gray_value, class_value in class_mapping.items():
    mask_classes[mask == gray_value] = class_value

# Plot the image and mask with legend
plt.figure(figsize=(12, 6))

# Image subplot
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")

# Mask subplot
plt.subplot(1, 2, 2)
plt.imshow(mask_classes, cmap=cmap)
plt.title("Segmentation Mask")
plt.axis("off")

# Add a legend
legend_handles = [plt.Line2D([0], [0], color=class_colors[idx], lw=4, label=class_name)
                  for idx, class_name in enumerate(class_names)]
plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

plt.tight_layout()
plt.show()



# import os
# import random
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
#
# # Directories containing images and masks
# image_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/Video01/Images/'
# mask_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/train/Video01/Labels_8_class'
#
# # Get list of image and mask filenames
# image_filenames = sorted(os.listdir(image_dir))
# mask_filenames = sorted(os.listdir(mask_dir))
#
# # Choose a random image and its corresponding mask
# random_index = random.randint(0, len(image_filenames) - 1)
# image_path = os.path.join(image_dir, image_filenames[random_index])
# mask_path = os.path.join(mask_dir, mask_filenames[random_index])
#
# # Load the image and mask
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
# # Define the class mapping (adjust based on your dataset)
# class_mapping = {
#     0: "Background",
#     1: "Pupil",
#     2: "Surgical Tape",
#     3: "Hand",
#     4: "Eye Retractors",
#     5: "Iris",
#     6: "Skin",
#     7: "Cornea",
#     8: "Instrument"
# }
#
# # Assign specific colors to each class
# class_colors = {
#     0: (0, 0, 0),         # Black for Background
#     1: (0, 0, 255),       # Red for Pupil
#     2: (0, 255, 0),       # Green for Surgical Tape
#     3: (255, 0, 0),       # Blue for Hand
#     4: (255, 255, 0),     # Yellow for Eye Retractors
#     5: (255, 0, 255),     # Magenta for Iris
#     6: (0, 255, 255),     # Cyan for Skin
#     7: (128, 0, 128),     # Purple for Cornea
#     8: (128, 128, 0)      # Olive for Instrument
# }
#
# # Create a colored mask for visualization
# colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
# for class_value, color in class_colors.items():
#     colored_mask[mask == class_value] = color
#
# # Prepare legend patches
# legend_patches = [Patch(color=np.array(color) / 255, label=label) for label, color in zip(class_mapping.values(), class_colors.values())]
#
# # Plot the image and mask with labels
# plt.figure(figsize=(15, 7))
#
# # Plot the original image
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Image")
# plt.axis("off")
#
# # Plot the mask with color and legend
# plt.subplot(1, 2, 2)
# plt.imshow(colored_mask)
# plt.title("Segmentation Mask")
# plt.axis("off")
# plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
# plt.tight_layout()
# plt.show()
