from scipy.stats import ttest_rel
import pickle
import os
import numpy as np
# num_classes = 8
# Load metric_dict_b from a file
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats, norm
from scipy.stats import wilcoxon

import pandas as pd
num_classes = 5
# class_names = {0: "Pupil", 1: "Surgical Tape", 2: "Hand", 3: "Eye Retractors", 4: "Iris",
#                5: "Skin", 6: "Cornea", 7: "Instruments"}
class_names = {1: "Iris", 2: "Pupil", 3: "Lens", 4: "Instruments"}
# Load metric_dict_a from a file
output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/Cataract1k/300_labeled_fully_supervised/full_supervised_experiment with base_lr=0.01, batch_size=24,labeld_num=300_300_labeled/output_ema/'
with open(os.path.join(output_dir, "metric_dict_sup_300_cataract.pkl"), 'rb') as file:
    metric_dict_a = pickle.load(file)

output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/experiment with labeled=300, unlabeled=40042_cataract1k_v5_300_labeled/output_ema/'
with open(os.path.join(output_dir, "metric_dict_sup_300_cataract_40k.pkl"), 'rb') as file:
    metric_dict_b = pickle.load(file)

from scipy.stats import ttest_rel

def perform_paired_ttest(metrics_a, metrics_b, metric_name="dice"):
    """
    Perform a paired t-test on the given metrics for two methods.
    Args:
        metrics_a: List of metrics for method A.
        metrics_b: List of metrics for method B.
        metric_name: Name of the metric (for display purposes).
    """
    t_stat, p_value = ttest_rel(metrics_a, metrics_b)
    print(f"Paired t-test for {metric_name}: t-statistic={t_stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"The improvement in {metric_name} is statistically significant (p < 0.05).")
    else:
        print(f"No significant improvement in {metric_name} (p >= 0.05).")

def calculate_confidence_interval(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    """
    Calculate confidence intervals for the difference between two means.
    """
    # Standard error of the difference
    se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))

    # Difference of means
    mean_diff = mean1 - mean2

    # Critical value (two-tailed)
    z = norm.ppf(1 - alpha / 2)

    # Confidence interval
    ci_lower = mean_diff - z * se_diff
    ci_upper = mean_diff + z * se_diff

    return mean_diff, ci_lower, ci_upper

def calculate_wilcoxon(x, y):
    stat, p_value = wilcoxon(x, y)
    return stat,p_value
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std
# Perform paired t-test for each class and metric
for key in range(num_classes):
    if key in metric_dict_a and key in metric_dict_b:
        metrics_a = [item[0] * 100  for item in metric_dict_a[key]]  # Dice scores for Method A
        metrics_b = [item[0] * 100 for item in metric_dict_b[key]]  # Dice scores for Method B
        if len(metrics_a) > 0 and len(metrics_b) > 0:
            # Step 2: Descriptive statistics
            baseline_mean = np.mean(metrics_a)
            new_method_mean = np.mean(metrics_b)
            baseline_std = np.std(metrics_a, ddof=1)
            new_method_std = np.std(metrics_b, ddof=1)
            print(f"Semi-supervised Mean {class_names[key]}: {baseline_mean:.2f}, Std Dev: {baseline_std:.2f}")
            print(f"Supervised Mean {class_names[key]}: {new_method_mean:.2f}, Std Dev: {new_method_std:.2f}")

            # # Perform independent t-test
            t_stat, p_value = ttest_ind_from_stats(
                mean1=baseline_mean, std1=baseline_std, nobs1=len(metrics_a),
                mean2=new_method_mean, std2=new_method_std, nobs2=len(metrics_b)
            )
            print(f"t_stat and p_value{class_names[key]}: {t_stat:.2f} : {p_value}")

            # Calculate effect size (Cohen's d)
            # effect_size = cohen_d(metrics_b, metrics_a)
            # print(f"Effect Size (Cohen's d) {class_names[key]}: {effect_size:.3f}")

            # pooled_std = np.sqrt(((len(metrics_a) - 1) * baseline_std ** 2 + (len(metrics_b) - 1) * new_method_std ** 2) / (
            #         len(metrics_a) +len(metrics_b) - 2))
            # cohen_d = (new_method_mean - baseline_mean) / pooled_std
            # print(f"cohen_d{class_names[key]}: {cohen_d:.2f}")

            # Calculate confidence interval
            # mean_diff, ci_lower, ci_upper = calculate_confidence_interval(
            #     mean1=baseline_mean, std1=baseline_std, n1=len(metrics_a),
            #     mean2=new_method_mean, std2=new_method_std, n2=len(metrics_b)
            # )
            # print(f"CI {class_names[key]}: {mean_diff:.2f} ({ci_lower:.2f}-{ci_upper:.2f})")
            # Step 3: Test assumptions
            # Normality test (Shapiro-Wilk)
            # _, p_baseline_normal = stats.shapiro(metrics_a)
            # _, p_new_method_normal = stats.shapiro(metrics_b)
            # print(f"Semi-supervised Normality p-value {class_names[key]}: {p_baseline_normal:.3f}")
            # print(f"Supervised Normality p-value {class_names[key]}: {p_new_method_normal:.3f}")

            # Variance homogeneity (Levene's test)
            # _, p_variance = stats.levene(metrics_a, metrics_b)
            # print(f"Variance Homogeneity p-value {class_names[key]}: {p_variance:.3f}")
            #
            # # Step 4: Perform appropriate statistical test
            # if p_baseline_normal > 0.05 and p_new_method_normal > 0.05:
            #     if p_variance > 0.05:
            #         # Use independent t-test (parametric)
            #         t_stat, p_value = stats.ttest_ind(metrics_b, metrics_a)
            #         test_used = "Independent t-test"
            #     else:
            #         # Use Welch's t-test
            #         t_stat, p_value = stats.ttest_ind(metrics_b, metrics_a, equal_var=False)
            #         test_used = "Welch's t-test"
            # else:
            #     # Use Mann-Whitney U test (non-parametric)
            #     t_stat, p_value = stats.mannwhitneyu(metrics_b, metrics_a)
            #     test_used = "Mann-Whitney U test"
            #
            # print(f"{test_used} Results: t-statistic {class_names[key]} = {t_stat:.3f}, p-value = {p_value:.3f}")
            #

            # Interpret the results
            stat,p_value_wilcoxon = calculate_wilcoxon(metrics_a, metrics_b)
            alpha = 0.05  # Significance level
            if p_value_wilcoxon < alpha:
                print(f"Significant improvement (reject H0) p-value {class_names[key]}: {p_value_wilcoxon}")
                # print("Significant improvement (reject H0)")
            else:
                print(f"No significant improvement (fail to reject H0){class_names[key]}")

            # Step 5: Effect size (Cohen's d)


            # # Step 6: Visualization
            # plt.figure(figsize=(10, 6))
            # plt.boxplot([metrics_b, metrics_a], labels_four_class=['Supervised', 'Semi-supervised'], patch_artist=True)
            # plt.title(f"Comparison of semi-supervised and supervised {class_names[key]}")
            # plt.ylabel('Scores')
            # plt.show()
#
#             # perform_paired_ttest(metrics_a, metrics_b, metric_name=f"Dice (Class {key})")
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
import pickle
import os

num_classes = 5
# class_names = {0: "Pupil", 1: "Surgical Tape", 2: "Hand", 3: "Eye Retractors", 4: "Iris",
#                5: "Skin", 6: "Cornea", 7: "Instruments"}
class_names = {1: "Iris", 2: "Pupil", 3: "Lens", 4: "Instruments"}
# Load metric_dict_a from a file
output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/Cataract1k/300_labeled_fully_supervised/full_supervised_experiment with base_lr=0.01, batch_size=24,labeld_num=300_300_labeled/output_ema/'
with open(os.path.join(output_dir, "metric_dict_sup_300_cataract.pkl"), 'rb') as file:
    metric_dict_a = pickle.load(file)

output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/experiment with labeled=300, unlabeled=40042_cataract1k_v5_300_labeled/output_ema/'
with open(os.path.join(output_dir, "metric_dict_sup_300_cataract_40k.pkl"), 'rb') as file:
    metric_dict_b = pickle.load(file)

# Collect metrics and plot box plots
boxplot_data = []
labels = []

for key in range(num_classes):
    if key in metric_dict_a and key in metric_dict_b:
        metrics_a = [item[0] * 100 for item in metric_dict_a[key]]  # Dice scores for Method A
        metrics_b = [item[0] * 100 for item in metric_dict_b[key]]  # Dice scores for Method B

        if len(metrics_a) > 0 or len(metrics_b) > 0:
            # Add data for box plots
            boxplot_data.append(metrics_a)  # Method A data
            boxplot_data.append(metrics_b)  # Method B data
            # Add labels_four_class for box plots
            labels.append(f"{class_names[key]} (Sup-Unet)")
            labels.append(f"{class_names[key]} (Semi-Sup MT)")

# Create box plots
plt.figure(figsize=(16, 8))
plt.boxplot(boxplot_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))

# Adjust x-ticks and labels_four_class
plt.xticks(rotation=45, fontsize=10)
plt.ylabel("Dice Score")
plt.title("Comparison of Dice Scores Across Classes for Supervised UNet and Semi-supervised MT")
#
# Add legend
# handles = [plt.Line2D([0], [0], color='lightblue', lw=4, label='Supervised Unet'),
#            plt.Line2D([0], [0], color='lightcoral', lw=4, label='Semi-supervised MT')]
# plt.legend(handles=handles, loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle
# import os
# # Step 1: Simulate data (replace this with your real data)
# np.random.seed(42)  # For reproducibility
# output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/experiment with labeled=600, unlabeled=10166_cataract1k_v3_600_labeled/output_1/'
# with open(os.path.join(output_dir, "metric_dict_demi_sup_600_cataract_10k.pkl"), 'rb') as file:
#     metric_dict_a = pickle.load(file)
#
# output_dir = '/Users/mahtab/Downloads/SSL4MIS/segmentation_project/model/Cataract1k/600_labeled_fully_supervised/full_supervised_experiment with base_lr=0.01, batch_size=24,labeld_num=600_600_labeled/output/'
# with open(os.path.join(output_dir, "metric_dict_sup_600_cataract_10k.pkl"), 'rb') as file:
#     metric_dict_b = pickle.load(file)
# baseline = np.random.normal(loc=70, scale=5, size=30)  # Baseline method
# new_method = np.random.normal(loc=75, scale=5, size=30)  # New segmentation method
#
# # Step 2: Descriptive statistics
# baseline_mean = np.mean(baseline)
# new_method_mean = np.mean(new_method)
# baseline_std = np.std(baseline, ddof=1)
# new_method_std = np.std(new_method, ddof=1)
# print(f"Baseline Mean: {baseline_mean:.2f}, Std Dev: {baseline_std:.2f}")
# print(f"New Method Mean: {new_method_mean:.2f}, Std Dev: {new_method_std:.2f}")
#
# # Step 3: Test assumptions
# # Normality test (Shapiro-Wilk)
# _, p_baseline_normal = stats.shapiro(baseline)
# _, p_new_method_normal = stats.shapiro(new_method)
# print(f"Baseline Normality p-value: {p_baseline_normal:.3f}")
# print(f"New Method Normality p-value: {p_new_method_normal:.3f}")
#
# # Variance homogeneity (Levene's test)
# _, p_variance = stats.levene(baseline, new_method)
# print(f"Variance Homogeneity p-value: {p_variance:.3f}")
#
# # Step 4: Perform appropriate statistical test
# if p_baseline_normal > 0.05 and p_new_method_normal > 0.05:
#     if p_variance > 0.05:
#         # Use independent t-test (parametric)
#         t_stat, p_value = stats.ttest_ind(baseline, new_method)
#         test_used = "Independent t-test"
#     else:
#         # Use Welch's t-test
#         t_stat, p_value = stats.ttest_ind(baseline, new_method, equal_var=False)
#         test_used = "Welch's t-test"
# else:
#     # Use Mann-Whitney U test (non-parametric)
#     t_stat, p_value = stats.mannwhitneyu(baseline, new_method)
#     test_used = "Mann-Whitney U test"
#
# print(f"{test_used} Results: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
#
# # Step 5: Effect size (Cohen's d)
# def cohen_d(x, y):
#     nx, ny = len(x), len(y)
#     dof = nx + ny - 2
#     pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
#     return (np.mean(x) - np.mean(y)) / pooled_std
#
# effect_size = cohen_d(baseline, new_method)
# print(f"Effect Size (Cohen's d): {effect_size:.3f}")
#
# # Step 6: Visualization
# plt.figure(figsize=(10, 6))
# plt.boxplot([baseline, new_method], labels_four_class=['Baseline', 'New Method'], patch_artist=True)
# plt.title('Comparison of Baseline and New Method')
# plt.ylabel('Scores')
# plt.show()
