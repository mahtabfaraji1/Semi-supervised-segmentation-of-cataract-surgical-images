
from networks.unet import UNet, UNetResnet


def net_factory(net_type, in_chns=3, class_num=5, backbone='resnet50'):
    """
    Instantiate a segmentation network by name.

    Args:
        net_type:   Network architecture identifier.
                    'unet'        -- plain UNet (no pretrained backbone).
                    'unet_resnet' -- UNet with ResNet50 backbone, ImageNet
                                     pretrained. Used in the paper.
        in_chns:    Number of input channels. Paper: 3 (RGB).
        class_num:  Number of output classes. Paper: 5.
        backbone:   Encoder backbone for unet_resnet. Paper: 'resnet50'.

    Returns:
        torch.nn.Module: Instantiated (and optionally pretrained) model.

    """
    if net_type == "unet":
        # Vanilla UNet -- used as supervised baseline architecture reference
        net = UNet(class_num, in_channels=in_chns, freeze_bn=False)

    elif net_type == "unet_resnet":
        # UNet with ResNet50 encoder, ImageNet pretrained.
        net = UNetResnet(
            class_num,
            in_channels=in_chns,
            backbone=backbone,
            pretrained=True,
            freeze_bn=False,
            freeze_backbone=False
        )

    else:
        raise ValueError(
            f"Unknown net_type: '{net_type}'. "
            f"Supported options: 'unet', 'unet_resnet'."
        )

    return net