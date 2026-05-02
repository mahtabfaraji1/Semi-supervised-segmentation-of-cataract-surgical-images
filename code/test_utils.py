
import os

import cv2
import numpy as np
from medpy import metric
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom
import torch



CLASS_COLORS = {
    0: (0,   0,   0),      # Background -- black
    1: (230, 159,   0),    # Iris        -- orange
    2: (86,  180, 233),    # Pupil       -- sky blue
    3: (0,   158, 115),    # IOL         -- bluish green
    4: (204, 121, 167),    # Instrument  -- reddish purple
}

CLASS_NAMES = {
    0: "Background",
    1: "Iris",
    2: "Pupil",
    3: "IOL",
    4: "Instrument",
}

# Metrics

def calculate_metric_percase(pred, gt):

    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dsc  = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return float(dsc), float(hd95)
    else:
        # Class not detected -- record as 0 (not filtered out)
        return 0.0, 0.0


# Colormap and visualization

def create_colormap():
    colors = [CLASS_COLORS[i] for i in range(5)]
    return ListedColormap(np.array(colors) / 255.0)


def apply_colormap(label, num_classes=5):
    cmap = create_colormap()
    colored = cmap(label / max(num_classes - 1, 1))   # normalize to [0,1]
    return (colored[:, :, :3] * 255).astype(np.uint8)


def save_prediction_image(prediction, metrics_per_class, output_path,
                          file_name, num_classes=5, image_size=None):
    if image_size is None:
        image_size = (prediction.shape[1], prediction.shape[0])

    pred_colored = apply_colormap(prediction, num_classes)
    pred_pil     = Image.fromarray(pred_colored).resize(image_size)
    pred_w, pred_h = pred_pil.size

    font_size   = max(12, pred_w // 18)
    line_height = font_size + 6
    bar_height  = 20 + line_height * len(metrics_per_class) + 20

    canvas = Image.new('RGB', (pred_w, pred_h + bar_height), color=(0, 0, 0))
    canvas.paste(pred_pil, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text_x = 20
    text_y = pred_h + 10
    for cls_id, (dsc, hd95) in sorted(metrics_per_class.items()):
        color    = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        draw.text(
            (text_x, text_y),
            f"{cls_name} | DSC: {dsc:.3f}  HD95: {hd95:.1f}",
            fill=color, font=font
        )
        text_y += line_height

    out_file = os.path.join(output_path, f"{file_name}_prediction.png")
    canvas.save(out_file)


def save_gt_image(label, output_path, file_name, num_classes=5):
    gt_dir = os.path.join(output_path, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    gt_colored = apply_colormap(label, num_classes)
    Image.fromarray(gt_colored).save(
        os.path.join(gt_dir, f"{file_name}_gt.png")
    )



def run_inference(image_path, label_path, net, output_dir, args, device):

    image = np.array(cv2.imread(image_path), dtype=np.uint8)
    label = np.array(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

    file_name = os.path.splitext(os.path.basename(image_path))[0]

    save_gt_image(label, output_dir, file_name, args.num_classes)

    h, w       = image.shape[0], image.shape[1]
    ph, pw     = args.patch_size
    resized    = zoom(image, (ph / h, pw / w, 1), order=0)

    inp = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
    inp = inp.squeeze(1).permute(0, 3, 1, 2)   # (1, C, H, W)

    net.eval()
    with torch.no_grad():
        logits     = net(inp)
        pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        prediction = pred_class.squeeze(0).cpu().numpy()

    prediction = zoom(prediction, (h / ph, w / pw), order=0)

    metrics_per_class = {}
    for cls_id in range(1, args.num_classes):
        pred_mask  = (prediction == cls_id).astype(np.uint8)
        label_mask = (label      == cls_id).astype(np.uint8)

        if label_mask.sum() > 0:
            metrics_per_class[cls_id] = calculate_metric_percase(pred_mask, label_mask)

    # Save prediction visualization
    os.makedirs(output_dir, exist_ok=True)
    save_prediction_image(
        prediction        = prediction,
        metrics_per_class = metrics_per_class,
        output_path       = output_dir,
        file_name         = file_name,
        num_classes       = args.num_classes,
        image_size        = (w, h)
    )

    return metrics_per_class



def aggregate_and_print(metric_dict, num_classes):

    print("\n" + "=" * 60)
    print(f"{'Class':<14} {'DSC mean':>10} {'DSC std':>10} "
          f"{'HD95 mean':>10} {'HD95 std':>10} {'N':>6}")
    print("=" * 60)

    summary = {}
    all_dsc, all_hd95 = [], []

    for cls_id in range(1, num_classes):
        values = metric_dict.get(cls_id, [])
        if len(values) == 0:
            continue

        dsc_vals  = [v[0] for v in values]
        hd95_vals = [v[1] for v in values]

        mean_dsc  = np.mean(dsc_vals)
        std_dsc   = np.std(dsc_vals)
        mean_hd95 = np.mean(hd95_vals)
        std_hd95  = np.std(hd95_vals)

        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        print(f"{cls_name:<14} {mean_dsc:>10.4f} {std_dsc:>10.4f} "
              f"{mean_hd95:>10.2f} {std_hd95:>10.2f} {len(dsc_vals):>6}")

        summary[cls_id] = {
            "dsc_mean": mean_dsc, "dsc_std": std_dsc,
            "hd95_mean": mean_hd95, "hd95_std": std_hd95,
            "n": len(dsc_vals)
        }
        all_dsc.extend(dsc_vals)
        all_hd95.extend(hd95_vals)

    if all_dsc:
        print("-" * 60)
        print(f"{'Macro avg':<14} {np.mean(all_dsc):>10.4f} {np.std(all_dsc):>10.4f} "
              f"{np.mean(all_hd95):>10.2f} {np.std(all_hd95):>10.2f} {len(all_dsc):>6}")
        summary["macro"] = {
            "dsc_mean": np.mean(all_dsc), "dsc_std": np.std(all_dsc),
            "hd95_mean": np.mean(all_hd95), "hd95_std": np.std(all_hd95)
        }
    print("=" * 60 + "\n")

    return summary
