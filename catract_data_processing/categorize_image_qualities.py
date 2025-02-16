import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import argparse


# ===========================
# 1. Define Image Quality Metrics
# ===========================

def get_image_resolution(image):
    """Returns the width and height of the image."""
    height, width = image.shape[:2]
    return width, height


def calculate_brightness(image):
    """Calculates the mean pixel intensity (brightness)."""
    return np.mean(image)


def calculate_contrast(image):
    """Calculates the standard deviation of pixel intensities (contrast)."""
    return np.std(image)


def calculate_focus(image):
    """Calculates the variance of the Laplacian (focus/sharpness)."""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


# ===========================
# 2. Process Image Directory
# ===========================

def process_directory(directory, label):
    """
    Processes all images in a directory and calculates image quality metrics.

    Parameters:
        directory (str): Path to the image directory.
        label (str): Label for the images ('Light' or 'Dark').

    Returns:
        List of dictionaries containing image metrics.
    """
    metrics = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(directory, filename)
            try:
                # Load image in grayscale
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Unable to read image {filepath}. Skipping.")
                    continue

                width, height = get_image_resolution(image)
                brightness = calculate_brightness(image)
                contrast = calculate_contrast(image)
                focus = calculate_focus(image)

                metrics.append({
                    'Image': filename,
                    'Label': label,
                    'Width': width,
                    'Height': height,
                    'Total_Pixels': width * height,
                    'Brightness': brightness,
                    'Contrast': contrast,
                    'Focus': focus
                })
            except Exception as e:
                print(f"Error processing image {filepath}: {e}")

    return metrics


# ===========================
# 3. Statistical Analysis
# ===========================

def perform_statistical_tests(df, metrics_list):
    """
    Performs t-tests on the specified metrics between two groups.

    Parameters:
        df (pd.DataFrame): DataFrame containing image metrics and labels_four_class.
        metrics_list (list): List of metric column names to test.

    Returns:
        pd.DataFrame: DataFrame containing t-test results.
    """
    results = []
    group1 = df[df['Label'] == metrics_list[0]]
    group2 = df[df['Label'] == metrics_list[1]]

    for metric in ['Brightness', 'Contrast', 'Focus']:
        stat, p = ttest_ind(
            group1[metric],
            group2[metric],
            equal_var=False  # Welch's t-test
        )
        results.append({
            'Metric': metric,
            'Group1 Mean': group1[metric].mean(),
            'Group2 Mean': group2[metric].mean(),
            't-Statistic': stat,
            'p-Value': p
        })

    return pd.DataFrame(results)


# ===========================
# 4. Visualization
# ===========================

def generate_boxplots(df, output_dir):
    """
    Generates box plots for each metric comparing two groups.

    Parameters:
        df (pd.DataFrame): DataFrame containing image metrics and labels_four_class.
        output_dir (str): Directory to save the plots.
    """
    metrics = ['Brightness', 'Contrast', 'Focus']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Label', y=metric, data=df)
        sns.stripplot(x='Label', y=metric, data=df, color='black', alpha=0.3)
        plt.title(f'{metric} Comparison by Iris Color')
        plt.savefig(os.path.join(output_dir, f'{metric}_boxplot.png'))
        plt.close()


def generate_histograms(df, output_dir):
    """
    Generates histograms for each metric comparing two groups.

    Parameters:
        df (pd.DataFrame): DataFrame containing image metrics and labels_four_class.
        output_dir (str): Directory to save the plots.
    """
    metrics = ['Brightness', 'Contrast', 'Focus']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=metric, hue='Label', kde=True, element='step', stat='density')
        plt.title(f'{metric} Distribution by Iris Color')
        plt.savefig(os.path.join(output_dir, f'{metric}_histogram.png'))
        plt.close()


# ===========================
# 5. Main Execution
# ===========================

def main(light_dir, dark_dir, output_dir):
    """
    Main function to process images, calculate metrics, perform statistical tests, and generate plots.

    Parameters:
        light_dir (str): Directory containing light iris images.
        dark_dir (str): Directory containing dark iris images.
        output_dir (str): Directory to save output plots and results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Processing light iris images...")
    light_metrics = process_directory(light_dir, 'Light')

    print("Processing dark iris images...")
    dark_metrics = process_directory(dark_dir, 'Dark')

    # Combine metrics
    all_metrics = light_metrics + dark_metrics
    df = pd.DataFrame(all_metrics)

    # Save metrics to CSV
    df.to_csv(os.path.join(output_dir, 'image_quality_metrics.csv'), index=False)
    print(f"Image quality metrics saved to {os.path.join(output_dir, 'image_quality_metrics.csv')}")

    # Perform statistical tests
    print("Performing statistical tests...")
    stats_results = perform_statistical_tests(df, ['Light', 'Dark'])
    stats_results.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    print(f"Statistical test results saved to {os.path.join(output_dir, 'statistical_tests.csv')}")
    print(stats_results)

    # Generate visualizations
    print("Generating box plots...")
    generate_boxplots(df, output_dir)

    print("Generating histograms...")
    generate_histograms(df, output_dir)

    print("Visualization plots saved to the output directory.")

    # Summary
    print("\n=== Summary ===")
    print(df.groupby('Label').agg({
        'Width': ['mean', 'std'],
        'Height': ['mean', 'std'],
        'Total_Pixels': ['mean', 'std'],
        'Brightness': ['mean', 'std'],
        'Contrast': ['mean', 'std'],
        'Focus': ['mean', 'std']
    }))

    print("\nAnalysis complete.")


# ===========================
# 6. Command-Line Interface
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Image Quality Metrics Between Two Image Sets.')
    parser.add_argument('--light_dir', type=str,
                        default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/bias_data_iris_color/light_iris_color/train/images')
    parser.add_argument('--dark_dir', type=str,
                        default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/bias_data_iris_color/dark_iris_color/train/images')
    parser.add_argument('--output_dir', type=str, default='/Users/mahtab/Downloads/SSL4MIS/segmentation_project/data/bias_data_iris_color/', help='Directory to save output metrics and plots.')

    args = parser.parse_args()

    main(args.light_dir, args.dark_dir, args.output_dir)
