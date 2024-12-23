import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to your log file
log_file = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/Experiment_labeled_50_labeled_1k_unlabeled_gpu_0_resnet_v1_50_labeled/unet_resnet/log.txt'

# Initialize lists to store epoch loss and validation metrics
epoch_losses = []
validation_metrics = []

# Regular expressions to match iteration, loss, and validation metrics lines
iteration_loss_pattern = r"iteration (\d+) : loss : ([\d.]+)"
validation_pattern = r"iteration (\d+) : mean_dice : ([\d.]+) mean_hd95 : ([\d.]+)"

# Read the log file and extract necessary data
with open(log_file, 'r') as file:
    for line in file:
        # Match loss per iteration
        loss_match = re.search(iteration_loss_pattern, line)
        if loss_match:
            iteration = int(loss_match.group(1))
            loss = float(loss_match.group(2))
            epoch = iteration // 5  # 5 iterations per epoch as per the log
            epoch_losses.append({'epoch': epoch, 'iteration': iteration, 'loss': loss})

        # Match validation metrics every 200 iterations
        validation_match = re.search(validation_pattern, line)
        if validation_match:
            iteration = int(validation_match.group(1))
            mean_dice = float(validation_match.group(2))
            mean_hd95 = float(validation_match.group(3))
            validation_metrics.append({'iteration': iteration, 'mean_dice': mean_dice, 'mean_hd95': mean_hd95})

# Convert lists to DataFrames for analysis
epoch_loss_df = pd.DataFrame(epoch_losses)
validation_metrics_df = pd.DataFrame(validation_metrics)

# Group by epoch to get average loss per epoch
avg_loss_per_epoch = epoch_loss_df.groupby('epoch')['loss'].mean().reset_index()

# Plotting average loss per epoch
plt.figure(figsize=(10, 5))
plt.plot(avg_loss_per_epoch['epoch'], avg_loss_per_epoch['loss'], marker='o', color='b', label='Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Loss per Epoch')
plt.legend()
plt.show()

# Plotting validation metrics every 200 iterations
plt.figure(figsize=(10, 5))
plt.plot(validation_metrics_df['iteration'], validation_metrics_df['mean_dice'], marker='o', color='g', label='Mean Dice')
plt.xlabel('Iteration')
plt.ylabel('Validation Metric')
plt.title('Validation Metrics Every 200 Iterations')
plt.legend()
# plt.figure(figsize=(10, 5))
# plt.plot(validation_metrics_df['iteration'], validation_metrics_df['mean_hd95'], marker='x', color='r', label='Mean HD95')
# plt.xlabel('Iteration')
# plt.ylabel('Validation Metric')
# plt.title('Validation Metrics Every 200 Iterations')
# plt.legend()
plt.show()
