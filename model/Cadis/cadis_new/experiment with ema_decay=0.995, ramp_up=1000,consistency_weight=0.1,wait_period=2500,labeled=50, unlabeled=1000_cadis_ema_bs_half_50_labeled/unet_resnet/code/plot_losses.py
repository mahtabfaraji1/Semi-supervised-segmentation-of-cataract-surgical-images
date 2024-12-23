import matplotlib.pyplot as plt

# Read the loss data from the .txt file
loss_values = []
loss_ce_values = []
loss_dice_values = []
iterations = []

# Assuming the data is stored in a file named 'loss_data.txt'
file_path = "/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/mean_teacher/30_perc_HP_Tuning/Experiment_labeled_30_per_473_labeled_e30000_consistency_1_v4_473_labeled/log.txt"

# Reading the data from the file
with open(file_path, 'r') as file:
    for line in file:
        if "iteration" in line:
            parts = line.strip().split(',')
            if len(parts) > 1:
                iter_num = int(parts[0].split(' ')[2])
                loss = float (parts[0].split(' ')[-1])
                print(iter_num)
                loss_ce = float(parts[1].split(':')[1].strip())
                loss_dice = float(parts[2].split(':')[1].strip())
                # loss_dice = float(parts[3].split(':')[1].strip())

                iterations.append(iter_num)
                loss_values.append(loss)
                loss_ce_values.append(loss_ce)
                loss_dice_values.append(loss_dice)

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, label='Total Loss', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.title('Total Loss')

plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_ce_values, label='Cross Entropy Loss', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.title('Cross Entropy Loss')

plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_dice_values, label='Dice Loss', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.title('Dice Loss')

# plt.xlabel('Iteration')
# plt.ylabel('Loss Value')
# plt.title('Losses over Iterations')
plt.legend()
plt.grid(True)
plt.show()
