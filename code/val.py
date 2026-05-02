import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def calculate_metric_percase(pred, gt):

    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dsc  = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dsc, hd95
    else:
        return 0.0, 0.0


def test_single_image(image, label, net, classes, device, patch_size=None):

    if patch_size is None:
        patch_size = [256, 256]

    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    h, w = image.shape[0], image.shape[1]
    if image.ndim == 3:
        resized = zoom(image, (patch_size[0] / h, patch_size[1] / w, 1), order=0)
    else:
        resized = zoom(image, (patch_size[0] / h, patch_size[1] / w), order=0)

    inp = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
    inp = inp.squeeze(1).permute(0, 3, 1, 2)   # (1, C, H, W)

    net.eval()
    with torch.no_grad():
        logits     = net(inp)
        pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        prediction = pred_class.squeeze(0).cpu().numpy()   # (patch_H, patch_W)

    prediction = zoom(prediction, (h / patch_size[0], w / patch_size[1]), order=0)

    metric_list = []
    for class_idx in range(1, classes):
        binary_pred  = (prediction == class_idx).astype(np.uint8)
        binary_label = (label      == class_idx).astype(np.uint8)

        if binary_pred.sum() > 0 and binary_label.sum() > 0:
            metric_list.append(calculate_metric_percase(binary_pred, binary_label))
        else:
            metric_list.append((0.0, 0.0))

    return metric_list