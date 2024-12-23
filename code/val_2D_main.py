import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from skimage.color import rgb2gray
# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     else:
#         return 0, 0
def calculate_metric_percase(pred, gt):
    # Ensure pred and gt are binary (0 or 1)
    pred = (pred > 0).astype(np.uint8)  # Convert to binary
    gt = (gt > 0).astype(np.uint8)  # Convert to binary

    # If there are any predicted positives, compute Dice and HD95
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)  # Dice coefficient
        hd95 = metric.binary.hd95(pred, gt)  # 95th percentile Hausdorff distance
        return dice, hd95
    else:
        # If pred is all zeros, return zero for both metrics
        return 0, 0
def test_single_image(image, label, net, classes,device,patch_size = [256, 256]):
    # if image.shape[1] == 3:
    #     image = rgb2gray(
    #         image.permute(1, 2, 0).cpu().numpy())
    #     image = torch.from_numpy(image).unsqueeze(0)  #
    #patch_size = [256, 256]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # image = rgb2gray(image)
    # x, y, _ = image.shape
    # image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=0)

    x, y = image.shape[0], image.shape[1]
    resized_image = zoom(image, (patch_size[0] / x, patch_size[1] / y,1), order=0)
    # Prepare the image as input for the network
    # input = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()
    input = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().to(device)
    input = input.squeeze(1)
    input = input.permute(0, 3, 1, 2)
    # Set the model to evaluation mode
    net.eval()


    # Run inference on the network
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        # Resize the output back to the original dimensions of the label
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

    # Compute metrics
    metric_list = []
    for i in range(1, classes):
        binary_prediction = (prediction == i).astype(np.uint8)  # Convert True/False to 1/0
        binary_label = (label == i).astype(np.uint8)  # Convert True/False to 1/0
        # Ensure binary arrays
        # assert binary_prediction.dtype == np.bool_, "Prediction array is not binary."
        # assert binary_label.dtype == np.bool_, "Label array is not binary."
        if np.sum(binary_prediction) > 0 and np.sum(binary_label) > 0:
            metric_list.append(calculate_metric_percase(binary_prediction, binary_label))
        else:
            metric_list.append((0, 0))
        # metric_list.append(calculate_metric_percase(binary_prediction, binary_label))
        # metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_image_gray(image, label, net, classes, device, patch_size=[256, 256]):
    # if image.shape[1] == 3:
    #     image = rgb2gray(
    #         image.permute(1, 2, 0).cpu().numpy())
    #     image = torch.from_numpy(image).unsqueeze(0)  #
    # patch_size = [256, 256]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # image = rgb2gray(image)

    x, y = image.shape
    resized_image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)

    # x, y = image.shape[0], image.shape[1]
    # resized_image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=0)
    # Prepare the image as input for the network
    # input = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()
    input = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float()

    input = input.squeeze(1)
    input = np.repeat(input[..., np.newaxis], 3, -1)
    # volume_batch = volume_batch.squeeze(1)
    input = input.permute(0, 3, 1, 2).to(device)

    # input = input.squeeze(1)
    # input = input.permute(0, 3, 1, 2)
    # Set the model to evaluation mode
    net.eval()

    # Run inference on the network
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        # Resize the output back to the original dimensions of the label
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

    # Compute metrics
    metric_list = []
    for i in range(1, classes):
        binary_prediction = (prediction == i).astype(np.uint8)  # Convert True/False to 1/0
        binary_label = (label == i).astype(np.uint8)  # Convert True/False to 1/0
        # Ensure binary arrays
        # assert binary_prediction.dtype == np.bool_, "Prediction array is not binary."
        # assert binary_label.dtype == np.bool_, "Label array is not binary."
        if np.sum(binary_prediction) > 0 and np.sum(binary_label) > 0:
            metric_list.append(calculate_metric_percase(binary_prediction, binary_label))
        else:
            metric_list.append((0, 0))
        # metric_list.append(calculate_metric_percase(binary_prediction, binary_label))
        # metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_image_cadis(image, label, net, classes,device,patch_size = [256, 256]):
    # if image.shape[1] == 3:
    #     image = rgb2gray(
    #         image.permute(1, 2, 0).cpu().numpy())
    #     image = torch.from_numpy(image).unsqueeze(0)  #
    #patch_size = [256, 256]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # image = rgb2gray(image)
    x, y = image.shape[0], image.shape[1]
    resized_image = zoom(image, (patch_size[0] / x, patch_size[1] / y,1), order=0)
    # Prepare the image as input for the network
    # input = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()
    input = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().to(device)
    input = input.squeeze(1)
    input = input.permute(0, 3, 1, 2)
    # Set the model to evaluation mode
    net.eval()


    # Run inference on the network
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        # Resize the output back to the original dimensions of the label
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

    # Compute metrics
    metric_list = []
    for i in range(0, classes):
        binary_prediction = (prediction == i).astype(np.uint8)  # Convert True/False to 1/0
        binary_label = (label == i).astype(np.uint8)  # Convert True/False to 1/0
        # Ensure binary arrays
        # assert binary_prediction.dtype == np.bool_, "Prediction array is not binary."
        # assert binary_label.dtype == np.bool_, "Label array is not binary."
        if np.sum(binary_prediction) > 0 and np.sum(binary_label) > 0:
            metric_list.append(calculate_metric_percase(binary_prediction, binary_label))
        else:
            metric_list.append((0, 0))
        # metric_list.append(calculate_metric_percase(binary_prediction, binary_label))
        # metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
# def test_single_volume(image, label, net, classes, patch_size=[256, 256],device=torch.device('cpu')):
#     image, label = image.squeeze(0).cpu().detach(
#     ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     image = color.rgb2gray(image)
#
#     prediction = np.zeros_like(label)
#
#     x, y = image.shape[0], image.shape[1]
#     input = torch.from_numpy(image).unsqueeze(
#         0).unsqueeze(0).float().to(device)
#     net.eval()
#
#     with torch.no_grad():
#         out = torch.argmax(torch.softmax(
#             net(input), dim=1), dim=1).squeeze(0)
#         out = out.cpu().detach().numpy()
#         pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#         prediction= pred
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(
#             prediction == i, label == i))
#
#
#     # for ind in range(image.shape[0]):
#     #     slice = image[ind, :, :]
#     #     x, y = slice.shape[0], slice.shape[1]
#     #     slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
#     #     input = torch.from_numpy(slice).unsqueeze(
#     #         0).unsqueeze(0).float().cuda()
#     #     net.eval()
#     #     with torch.no_grad():
#     #         out = torch.argmax(torch.softmax(
#     #             net(input), dim=1), dim=1).squeeze(0)
#     #         out = out.cpu().detach().numpy()
#     #         pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#     #         prediction[ind] = pred
#     # metric_list = []
#     # for i in range(1, classes):
#     #     metric_list.append(calculate_metric_percase(
#     #         prediction == i, label == i))
#     return metric_list
def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
