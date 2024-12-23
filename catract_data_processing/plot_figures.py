# import matplotlib.pyplot as plt
# import numpy as np
#
# # Create some data for x and y (U-Net and MeanTeacher)
# x = [50,200,400,600,800,1000]
#
# # U-Net data (DCS) with standard deviations
# y1_unet = np.array([0.56,0.81,0.86,0.87,0.87,0.89])  # cornea
# y1_unet_std = np.array([0.13,0.08,0.09,0.08,0.08,0.08])
# y2_unet = np.array([0.67,0.78,0.85,0.88,0.89,0.89])  # pupil
# y2_unet_std = np.array([0.23,0.23,0.15,0.12,0.1,0.12])
# y3_unet = np.array([0.66,0.79,0.82,0.86,0.89,0.90])  # lens
# y3_unet_std = np.array([0.16,0.12,0.15,0.12,0.09,0.06])
# y4_unet = np.array([0.34,0.41,0.65,0.70,0.73,0.75])  # instrument
# y4_unet_std = np.array([0.16,0.17,0.18,0.17,0.17,0.16])
#
# y5_unet = np.array([0.55,0.69,0.79,0.82,0.84,0.85])  # instrument
#
# # MeanTeacher (MT) data (DCS) with standard deviations
# y1_mt = np.array([0.67,0.82,0.87,0.88,0.88,0.89])  # cornea
# y1_mt_std = np.array([0.11,0.08,0.08,0.08,0.08,0.08])
# y2_mt = np.array([0.70,0.79,0.87,0.89,0.90,0.90])  # pupil
# y2_mt_std = np.array([0.24,0.23,0.12,0.11,0.1,0.1])
# y3_mt = np.array([0.66,0.79,0.87,0.87,0.89,0.89])  # lens
# y3_mt_std = np.array([0.19,0.12,0.11,0.12,0.07,0.06])
# y4_mt = np.array([0.34,0.46,0.66,0.73,0.76,0.76])  # instrument
# y4_mt_std = np.array([0.15,0.17,0.16,0.17,0.15,0.16])
#
# y5_mt = np.array([0.59,0.71,0.81,0.84,0.85,0.85])  # Average
#
# # U-Net data (DCS) with standard deviations
#
# # y1_unet = np.array([54.65,25.64,16.87,16.94,14.81,13.35])  # cornea
# # # y1_unet_std = np.array([])
# # y2_unet = np.array([90.47,46.73,25.95,18.64,18.91,17.04])  # pupil
# # # y2_unet_std = np.array([])
# # y3_unet = np.array([70.09,52.38,40.13,35.53,29.93,27.06])  # lens
# # # y3_unet_std = np.array([])
# # y4_unet = np.array([270.03,197.25,88.48,55.61,57.16,45.57])  # instrument
# # # y4_unet_std = np.array([127.73])
# #
# # # MeanTeacher (MT) data (DCS) with standard deviations
# # y1_mt = np.array([46.09,25.88,16.11,14.10,12.85,12.95])  # cornea
# # # y1_mt_std = np.array([])
# # y2_mt = np.array([82.55,37.84,20.02,16.32,15.57,16.85])  # pupil
# # # y2_mt_std = np.array([])
# # y3_mt = np.array([65.32,48.77,35.0,33.97,28.95,26.42])  # lens
# # # y3_mt_std = np.array([]) 33.97
# # y4_mt = np.array([249.87,179.16,75.44,50.90,46.64,48.24])  # instrument
# # # y4_mt_std = np.array([])
# # # Create exponential values for x-axis
# # # x_exp = np.log10(x)
#
# # Create a 2x2 grid of plots with markers for both U-Net and MT
# fig, axs = plt.subplots(2, 3, figsize=(12, 10))
#
# # Define colors and markers for consistency
# unet_color = '#1f77b4'  # Blue
# mt_color = '#ff7f0e'    # Orange
# unet_marker = 'o'
# mt_marker = 's'
#
# # Shared customization options
# plot_params = {
#     'linestyle': '-',
#     'markersize': 8,
#     'elinewidth': 1.5,
#     'capthick': 1.5,
#     'capsize': 5,
#     'alpha': 0.8
# }
#
# # Plot U-Net and MT data with error bars
# # axs[0, 0].errorbar(x, y1_unet, yerr=y1_unet_std, fmt=unet_marker, color=unet_color, label='U-Net', **plot_params)
# axs[0, 0].errorbar(x, y1_unet, fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
# axs[0, 0].errorbar(x, y1_mt, fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
# axs[0, 0].set_title('Iris', fontsize=14)
# axs[0, 0].set_xlabel('Number of Labeled Data', fontsize=12)
# axs[0, 0].set_ylabel('DCS', fontsize=12)
# axs[0, 0].grid(True, linestyle='--', alpha=0.7)
#
# axs[0, 1].errorbar(x, y2_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
# axs[0, 1].errorbar(x, y2_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
# axs[0, 1].set_title('Pupil', fontsize=14)
# axs[0, 1].set_xlabel('Number of Labeled Data', fontsize=12)
# axs[0, 1].set_ylabel('DCS', fontsize=12)
# axs[0, 1].grid(True, linestyle='--', alpha=0.7)
#
# axs[0, 2].errorbar(x, y3_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
# axs[0, 2].errorbar(x, y3_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
# axs[0, 2].set_title('Lens', fontsize=14)
# axs[0, 2].set_xlabel('Number of Labeled Data', fontsize=12)
# axs[0, 2].set_ylabel('DCS', fontsize=12)
# axs[0, 2].grid(True, linestyle='--', alpha=0.7)
#
# axs[1, 0].errorbar(x, y4_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
# axs[1, 0].errorbar(x, y4_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
# axs[1, 0].set_title('Instrument', fontsize=14)
# axs[1, 0].set_xlabel('Number of Labeled Data', fontsize=12)
# axs[1, 0].set_ylabel('DCS', fontsize=12)
# axs[1, 0].grid(True, linestyle='--', alpha=0.7)
#
#
# axs[1, 1].errorbar(x, y5_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
# axs[1, 1].errorbar(x, y5_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
# axs[1, 1].set_title('Average', fontsize=14)
# axs[1, 1].set_xlabel('Number of Labeled Data', fontsize=12)
# axs[1, 1].set_ylabel('DCS', fontsize=12)
# axs[1, 1].grid(True, linestyle='--', alpha=0.7)
# # Add legends to all subplots
# for ax in axs.flat:
#     ax.legend(fontsize=10, frameon=True, shadow=True)
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Save and display the plot
# fig.savefig('DSC_Exponential_Scale_with_Std_Pretty.png', dpi=300)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Create some data for x and y (U-Net and MeanTeacher)
x = [50, 200, 400, 600, 800, 1000]

# U-Net data (DCS) with standard deviations
y1_unet = np.array([0.56, 0.81, 0.86, 0.87, 0.87, 0.89])  # cornea
y1_unet_std = np.array([0.13, 0.08, 0.09, 0.08, 0.08, 0.08])
y2_unet = np.array([0.67, 0.78, 0.85, 0.88, 0.89, 0.89])  # pupil
y2_unet_std = np.array([0.23, 0.23, 0.15, 0.12, 0.1, 0.12])
y3_unet = np.array([0.66, 0.79, 0.82, 0.86, 0.89, 0.90])  # lens
y3_unet_std = np.array([0.16, 0.12, 0.15, 0.12, 0.09, 0.06])
y4_unet = np.array([0.34, 0.41, 0.65, 0.70, 0.73, 0.75])  # instrument
y4_unet_std = np.array([0.16, 0.17, 0.18, 0.17, 0.17, 0.16])
y5_unet = np.array([0.55, 0.69, 0.79, 0.82, 0.84, 0.85])  # Average

# MeanTeacher (MT) data (DCS) with standard deviations
y1_mt = np.array([0.67, 0.82, 0.87, 0.88, 0.88, 0.89])  # cornea
y1_mt_std = np.array([0.11, 0.08, 0.08, 0.08, 0.08, 0.08])
y2_mt = np.array([0.70, 0.79, 0.87, 0.89, 0.90, 0.90])  # pupil
y2_mt_std = np.array([0.24, 0.23, 0.12, 0.11, 0.1, 0.1])
y3_mt = np.array([0.66, 0.79, 0.87, 0.87, 0.89, 0.89])  # lens
y3_mt_std = np.array([0.19, 0.12, 0.11, 0.12, 0.07, 0.06])
y4_mt = np.array([0.34, 0.46, 0.66, 0.73, 0.76, 0.76])  # instrument
y4_mt_std = np.array([0.15, 0.17, 0.16, 0.17, 0.15, 0.16])
y5_mt = np.array([0.59, 0.71, 0.81, 0.84, 0.85, 0.85])  # Average

# Create a 2x3 grid of plots
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

# Plot titles
titles = ['Iris', 'Pupil', 'Lens', 'Instrument', 'Average']

# Data for plots
unet_data = [y1_unet, y2_unet, y3_unet, y4_unet, y5_unet]
mt_data = [y1_mt, y2_mt, y3_mt, y4_mt, y5_mt]
unet_std = [y1_unet_std, y2_unet_std, y3_unet_std, y4_unet_std]
mt_std = [y1_mt_std, y2_mt_std, y3_mt_std, y4_mt_std]

# Colors and markers
colors = {'U-Net': '#1f77b4', 'MT': '#ff7f0e'}
markers = {'U-Net': 'o', 'MT': 's'}

# Shared plot parameters
plot_params = {
    'linestyle': '-',
    'markersize': 6,
    'elinewidth': 1,
    'capthick': 1.2,
    'capsize': 4,
    'alpha': 0.9
}

# Loop over plots
for i, ax in enumerate(axs.flat):
    if i < 5:  # For the first 5 plots
        ax.errorbar(x, unet_data[i], fmt=markers['U-Net'], color=colors['U-Net'], label='Supervised U-Net', **plot_params)
        ax.errorbar(x, mt_data[i], fmt=markers['MT'], color=colors['MT'], label='Semi-supervised MT', **plot_params)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('Number of Labeled Data', fontsize=12)
        ax.set_ylabel('DCS', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, frameon=True)
    else:  # Leave the 6th plot empty
        ax.axis('off')

# Adjust layout and save
plt.tight_layout()
fig.savefig('DSC_Exponential_Scale_with_Std_Pretty.png', dpi=300)
plt.show()
