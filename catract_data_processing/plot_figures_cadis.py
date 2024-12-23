import matplotlib.pyplot as plt
import numpy as np

# Create some data for x and y (U-Net and MeanTeacher)
x = [50,200,400,600]

# U-Net data (DCS) with standard deviations
y1_unet = np.array([0.60,0.92,0.92,0.92])  # Pupil

y2_unet = np.array([0.31,0.52,0.76,0.79])  # Surgical Tape

y3_unet = np.array([0.06,0.36,0.38,0.52])  # Hand

y4_unet = np.array([None,0.46,0.56,0.63])  # Eye Retractors

y5_unet = np.array([0.24,0.70,0.75,0.80])  # Iris

y6_unet = np.array([0.09,0.61,0.66,0.68])  # Skin

y7_unet = np.array([0.80,0.91,0.93,0.94])  # Cornea

y8_unet = np.array([0.26,0.50,0.60,0.63])  # Instruments

y9_unet = np.array([0.33,0.62,0.69,0.74])  # Avergae



# MeanTeacher (MT) data (DCS) with standard deviations
y1_mt = np.array([0.87,0.93,0.93,0.96])  # Pupil

y2_mt = np.array([0.38,0.53,0.78,0.81])  # Surgical Tape

y3_mt = np.array([0.22,0.45,0.46,0.48])  # Hand

y4_mt = np.array([None,0.46,0.54,0.62])  # Eye Refractors

y5_mt = np.array([0.36,0.78,0.81,0.85])  # Iris

y6_mt = np.array([0.30,0.60,0.68,0.70])  # Skin

y7_mt = np.array([0.84,0.92,0.94,0.94])  # Cornea

y8_mt = np.array([0.30,0.53,0.63,0.67])  # Instruments

y9_mt = np.array([0.46,0.65,0.72,0.75])  # Avergae
# U-Net data (DCS) with standard deviations

# y1_unet = np.array([183.97,68.98,73.15,46.4])  # Pupil
#
# y2_unet = np.array([389.22,316.83,176.4,143.66])  # Surgical Tape
#
# y3_unet = np.array([193.74,153.95,194.62,188.96])  # Hand
#
# y4_unet = np.array([None,254.38,184.08,192.36])  # Eye Retractors
#
# y5_unet = np.array([232.54,53.87,39.18,31.38])  # Iris
#
# y6_unet = np.array([421.70,485.69,426.71,455.73])  # Skin
#
# y7_unet = np.array([141.0,76.86,48.15,43.10])  # Cornea
#
# y8_unet = np.array([356.79,277.89,170.2,146.47])  # Instruments
#
#
#
# # MeanTeacher (MT) data (DCS) with standard deviations
# y1_mt = np.array([215.41,81.23,49.02,46.54])  # Pupil
#
# y2_mt = np.array([322.4,284.03,142.91,156.48])  # Surgical Tape
#
# y3_mt = np.array([197.69,198.91,166.96,235.17])  # Hand
#
# y4_mt = np.array([None,274.4,203.86,148.68])  # Eye Refractors
#
# y5_mt = np.array([120.16,40.62,37.39,30.66])  # Iris
#
# y6_mt = np.array([437.61,474.92,422.67,426.04])  # Skin
#
# y7_mt = np.array([99.76,72.83,55.22,45.55])  # Cornea
#
# y8_mt = np.array([353.92,257.5,176.89,143.85])  # Instruments

# Create a 2x2 grid of plots with markers for both U-Net and MT
fig, axs = plt.subplots(3, 3, figsize=(12, 10))

# Define colors and markers for consistency
unet_color = '#1f77b4'  # Blue
mt_color = '#ff7f0e'    # Orange
unet_marker = 'o'
mt_marker = 's'

# Shared customization options
plot_params = {
    'linestyle': '-',
    'markersize': 8,
    'elinewidth': 1.5,
    'capthick': 1.5,
    'capsize': 5,
    'alpha': 0.8
}

# Plot U-Net and MT data with error bars
# axs[0, 0].errorbar(x, y1_unet, yerr=y1_unet_std, fmt=unet_marker, color=unet_color, label='U-Net', **plot_params)
axs[0, 0].errorbar(x, y1_unet, fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[0, 0].errorbar(x, y1_mt, fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[0, 0].set_title('Pupil', fontsize=14)
axs[0, 0].set_xlabel('Number of Labeled Data', fontsize=12)
axs[0, 0].set_ylabel('DCS', fontsize=12)
axs[0, 0].grid(True, linestyle='--', alpha=0.7)

axs[0, 1].errorbar(x, y2_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[0, 1].errorbar(x, y2_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[0, 1].set_title('Surgical Tape', fontsize=14)
axs[0, 1].set_xlabel('Number of Labeled Data', fontsize=12)
axs[0, 1].set_ylabel('DCS', fontsize=12)
axs[0, 1].grid(True, linestyle='--', alpha=0.7)


axs[0, 2].errorbar(x, y3_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[0, 2].errorbar(x, y3_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[0, 2].set_title('Hand', fontsize=14)
axs[0, 2].set_xlabel('Number of Labeled Data', fontsize=12)
axs[0, 2].set_ylabel('DCS', fontsize=12)
axs[0, 2].grid(True, linestyle='--', alpha=0.7)

axs[1, 0].errorbar(x, y4_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[1, 0].errorbar(x, y4_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[1, 0].set_title('Eye Refractors', fontsize=14)
axs[1, 0].set_xlabel('Number of Labeled Data', fontsize=12)
axs[1, 0].set_ylabel('DCS', fontsize=12)
axs[1, 0].grid(True, linestyle='--', alpha=0.7)


axs[1, 1].errorbar(x, y5_unet, fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[1, 1].errorbar(x, y5_mt, fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[1, 1].set_title('Iris', fontsize=14)
axs[1, 1].set_xlabel('Number of Labeled Data', fontsize=12)
axs[1, 1].set_ylabel('DCS', fontsize=12)
axs[1, 1].grid(True, linestyle='--', alpha=0.7)

axs[1, 2].errorbar(x, y6_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[1, 2].errorbar(x, y6_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[1, 2].set_title('Skin', fontsize=14)
axs[1, 2].set_xlabel('Number of Labeled Data', fontsize=12)
axs[1, 2].set_ylabel('DCS', fontsize=12)
axs[1, 2].grid(True, linestyle='--', alpha=0.7)


axs[2, 0].errorbar(x, y7_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[2, 0].errorbar(x, y7_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[2, 0].set_title('Cornea', fontsize=14)
axs[2, 0].set_xlabel('Number of Labeled Data', fontsize=12)
axs[2, 0].set_ylabel('DCS', fontsize=12)
axs[2, 0].grid(True, linestyle='--', alpha=0.7)

axs[2, 1].errorbar(x, y8_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[2, 1].errorbar(x, y8_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[2, 1].set_title('Instruments', fontsize=14)
axs[2, 1].set_xlabel('Number of Labeled Data', fontsize=12)
axs[2, 1].set_ylabel('DCS', fontsize=12)
axs[2, 1].grid(True, linestyle='--', alpha=0.7)


axs[2, 2].errorbar(x, y9_unet,  fmt=unet_marker, color=unet_color, label='Supervised U-Net', **plot_params)
axs[2, 2].errorbar(x, y9_mt,  fmt=mt_marker, color=mt_color, label='Semi-supervised MT', **plot_params)
axs[2, 2].set_title('Average', fontsize=14)
axs[2, 2].set_xlabel('Number of Labeled Data', fontsize=12)
axs[2, 2].set_ylabel('DCS', fontsize=12)
axs[2, 2].grid(True, linestyle='--', alpha=0.7)
# Add legends to all subplots
for ax in axs.flat:
    ax.legend(fontsize=10, frameon=True, shadow=True)

# Adjust layout for better spacing
plt.tight_layout()

# Save and display the plot
fig.savefig('DSC_Cadis_Exponential_Scale_with_Std_Pretty.png', dpi=300)
plt.show()
