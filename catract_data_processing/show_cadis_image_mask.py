import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define label mapping: map specific mask labels to target classes
label_mapping = {
    0: 2,  # Pupil
    4: 1  # iris
}
# Define additional labels for Instruments
instruments_labels = [label for label in range(33) if label not in {0,1, 2, 3,4, 5, 6}]
label_mapping.update({label: 4 for label in instruments_labels})


def map_labels(mask, label_mapping):
    """
    Map mask labels based on the given mapping dictionary.
    Args:
        mask (np.array): The input mask with fine-grained labels.
        label_mapping (dict): The mapping from fine-grained labels to target classes.
    Returns:
        np.array: The mapped mask with target classes.
    """
    mapped_mask = np.vectorize(lambda x: label_mapping.get(x, 0))(mask)
    return mapped_mask


def create_custom_colormap():
    """
    Create a custom color map for displaying the classes in the mask.
    Returns:
        ListedColormap: The custom color map.
    """
    colors = [
        (0, 0, 0),  # Background (black)
        (255, 0, 0),  # Cornea (red)
        (0, 255, 0),  # Pupil (green)
        (0, 0, 255),  # Lens (blue, if needed)
        (255, 255, 0),  # Instruments (yellow)
    ]
    return ListedColormap(np.array(colors) / 255.0)


def display_image_and_mask(image, mask, mapped_mask):
    """
    Display the original image, original mask, and mapped mask side-by-side.
    Args:
        image (np.array): The original RGB image.
        mask (np.array): The original mask with fine-grained labels.
        mapped_mask (np.array): The mapped mask with target classes.
    """
    cmap = create_custom_colormap()

    # Display the original image, original mask, and mapped mask
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Original Mask")
    axs[1].axis("off")

    axs[2].imshow(mapped_mask, cmap=cmap)
    axs[2].set_title("Mapped Mask")
    axs[2].axis("off")

    plt.show()


# Load your image and mask
# Replace 'path_to_image' and 'path_to_mask' with actual file paths
image = cv2.imread('/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/Video02/Images/Video2_frame000440.png')
mask = cv2.imread('/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/CaDISv2/test/Video02/Labels/Video2_frame000440.png', cv2.IMREAD_GRAYSCALE)

# Map the mask labels
mapped_mask = map_labels(mask, label_mapping)

# Display the image and masks
display_image_and_mask(image, mask, mapped_mask)
