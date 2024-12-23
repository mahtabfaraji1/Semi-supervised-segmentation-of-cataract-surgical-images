import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the learning curve function
def learning_curve(x, b1, b2):
    return 1.0 + b1 * (x ** b2)

# Training data sizes and mean segmentation metric values (e.g., mean Dice coefficient)
train_sizes = np.array([50, 100, 150, 200, 250, 900])
mean_metrics = np.array([0.59, 0.59, 0.62,0.69, 0.72, 0.85])  # Replace with your mean metric values
std_devs = np.array([0.17, 0.18, 0.14, 0.15, 0.14, 0.11])  # Replace with std devs for each data size
# Fit the learning curve model to the data, using std deviations for sigma
params, covariance = curve_fit(learning_curve, train_sizes, mean_metrics, sigma=std_devs)

# Extract the parameters
b1, b2 = params

# Predicted mean metric values for a range of training sizes
train_sizes_extended = np.linspace(50, 50000, 100)
predicted_metrics = learning_curve(train_sizes_extended, b1, b2)

# Set a target metric value (e.g., 95 for Dice coefficient)
desired_metric = 0.95
required_train_size = ((desired_metric - 1.00) / b1) ** (1 / b2)
print(f"Estimated training size needed to achieve {desired_metric} target metric: {required_train_size:.0f} images")


test_sample = 1200
test_sample_predicted_metrics = learning_curve(test_sample, b1, b2)
true_value = 0.86

print(f"Estimated test sample for sample num: {test_sample} is {test_sample_predicted_metrics}% true value: {true_value}")


# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, mean_metrics, 'o', label="Observed Mean Metric", color='blue')
plt.plot(train_sizes_extended, predicted_metrics, '-', label="Learning Curve Fit", color='red')
plt.axhline(y=desired_metric, color='green', linestyle='--', label=f"Target Value: {desired_metric} ")
plt.axvline(x=required_train_size, color='purple', linestyle='--', label=f"Estimated Data Size: {required_train_size:.0f}")

# # Plot test sample predicted and true values
# plt.plot(test_sample, test_sample_predicted_metrics, 'o', color='orange', label="Test Sample Predicted")
# plt.plot(test_sample, true_value, 'o', color='cyan', label="Test Sample True")

# Labels and legend
plt.xlabel("Training Data Size")
plt.ylabel("DSC")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.savefig("how_much_data.png")
plt.show()
