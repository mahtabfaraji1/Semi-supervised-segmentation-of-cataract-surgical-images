import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Multi-class Dice loss.

    Computes the mean Dice loss across all classes (including background).
    Each class is treated as a binary segmentation problem using one-hot
    encoding of the ground-truth label.

    Dice loss per class (smooth=1e-5 for numerical stability):
        L_Dice_c = 1 - (2 * sum(pred_c * target_c) + smooth)
                       / (sum(pred_c^2) + sum(target_c^2) + smooth)

    Final loss = mean over all classes.

    Args:
        n_classes: Number of segmentation classes. Paper: 5.
    """

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """Convert integer label map to one-hot float tensor."""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i * torch.ones_like(input_tensor))
            tensor_list.append(temp_prob)
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        """Binary Dice loss between predicted probability map and binary target."""
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum     = torch.sum(target * target)
        z_sum     = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Compute mean Dice loss across all classes.

        Args:
            inputs:  Predicted softmax probabilities, shape (B, C, H, W).
                     If softmax=True, raw logits are accepted and softmax
                     is applied internally.
            target:  Ground truth label map, shape (B, 1, H, W), integer.
            weight:  Per-class loss weights. Defaults to uniform (all 1.0).
            softmax: If True, apply softmax to inputs before computing loss.

        Returns:
            torch.Tensor: Scalar Dice loss averaged over classes.
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1.0] * self.n_classes

        assert inputs.size() == target.size(), \
            f"Shape mismatch: inputs {inputs.size()} vs target {target.size()}"

        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]

        return loss / self.n_classes