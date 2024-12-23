# import torch
# import matplotlib.pyplot as plt
#
# # Assuming `volume_batch` is the tensor from your code (shape: [24, 1, 256, 256])
# # Extract one image from the batch (e.g., the first image)
# image = ema_inputs_byte[0,:, :, :]  # Select the first image (index 0) and remove channel dimension
# # npoisy_tensor_1 =normalized_tensor.squeeze(1)
#
# # image = ema_inputs[8,:, :, :]  # Select the first image (index 0) and remove channel dimension
# # image =image.permute(2, 1, 0)
# # Convert the tensor to a NumPy array for plotting
# image_np = image.cpu().numpy() if torch.is_tensor(image) else image
#
# # Plot the image
# plt.figure(figsize=(6, 6))
# plt.imshow(image_np)  # Assuming it's grayscale
# plt.title("Image from Volume Batch")
# plt.colorbar()
# plt.axis('off')
# plt.show()
#
# ema_inputs_byte = ema_inputs.byte()
# image = ema_inputs_byte[10,:,:,:]
# image_np =image.permute(1, 2, 0).cpu().numpy()
# plt.imshow(image_np)  # Assuming it's grayscale
import matplotlib.pyplot as plt
import torch


def plot_tensor_image(tensor_batch, index=0):
    """
    Plots a single image from a tensor batch.

    Args:
        tensor_batch (torch.Tensor): Tensor of images with shape (N, C, H, W).
        index (int): Index of the image in the batch to plot.
    """
    # Extract the image at the specified index
    index = 0
    image = image_batch[index]

    # Permute the dimensions from (C, H, W) to (H, W, C) for plotting
    # image = image.permute(1, 2, 0).cpu().numpy()
    image_np = image.cpu().numpy()

    # Normalize the image to the range [0, 1]
    image_np = image_np / 255.0

    # Plot the image
    plt.imshow(image_np)
    plt.axis('off')
    # plt.title(f"Image at Index {index}")
    plt.show()


# Example usage
plot_tensor_image(ema_inputs, index=0)  # Change the index to view other images in the batch
