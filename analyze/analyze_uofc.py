import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
data = pd.read_csv('../results/results_uofc.csv')

# Separate Gaussian and non-Gaussian datasets
gaussian_data = data[data['dataset_type'] == 'gaussian']
non_gaussian_data = data[data['dataset_type'] != 'gaussian']


def analyze_gaussian_data(df):
    # Group by n_features and cluster_std
    grouped = df.groupby(['n_features', 'cluster_std'])

    # Calculate mean accuracy for each group
    mean_accuracy = grouped['accuracy'].mean().unstack()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_accuracy, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title('Mean Accuracy by Number of Features and Cluster Standard Deviation')
    plt.xlabel('Cluster Standard Deviation')
    plt.ylabel('Number of Features')
    plt.savefig('gaussian_heatmap.png')
    plt.close()

    return mean_accuracy


def analyze_cluster_estimation(df):
    # Calculate the percentage of correct estimations
    df['correct_estimation'] = (df['n_centers'] == df['estimated_n_centers']).astype(int)
    estimation_accuracy = df.groupby(['n_features', 'cluster_std'])['correct_estimation'].mean().unstack()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(estimation_accuracy, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Cluster Number Estimation Accuracy')
    plt.xlabel('Cluster Standard Deviation')
    plt.ylabel('Number of Features')
    plt.savefig('cluster_estimation_heatmap.png')
    plt.close()

    return estimation_accuracy


def analyze_non_gaussian_data(df):
    results = df.groupby('dataset_type').agg({
        'n_features': 'first',
        'n_centers': 'first',
        'estimated_n_centers': 'first',
        'accuracy': 'first'
    })
    return results


def plot_accuracy_distribution(gaussian_accuracies, non_gaussian_accuracies):
    plt.figure(figsize=(10, 6))
    sns.histplot(gaussian_accuracies, kde=True, label='Gaussian')
    sns.histplot(non_gaussian_accuracies, kde=True, label='Non-Gaussian')
    plt.title('Distribution of Accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('accuracy_distribution.png')
    plt.close()


# Analyze Gaussian data
gaussian_results = analyze_gaussian_data(gaussian_data)
cluster_estimation_results = analyze_cluster_estimation(gaussian_data)

# Analyze non-Gaussian data
non_gaussian_results = analyze_non_gaussian_data(non_gaussian_data)

# Plot accuracy distribution
plot_accuracy_distribution(gaussian_data['accuracy'], non_gaussian_data['accuracy'])

# Perform t-test between Gaussian and non-Gaussian accuracies
gaussian_accuracies = gaussian_data['accuracy']
non_gaussian_accuracies = non_gaussian_data['accuracy']
t_stat, p_value = stats.ttest_ind(gaussian_accuracies, non_gaussian_accuracies)

print("Gaussian Data Analysis:")
print(gaussian_results)
print("\nCluster Estimation Accuracy:")
print(cluster_estimation_results)
print("\nNon-Gaussian Data Analysis:")
print(non_gaussian_results)

print(f"\nt-test results:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Calculate overall statistics
overall_mean = data['accuracy'].mean()
overall_std = data['accuracy'].std()
gaussian_mean = gaussian_accuracies.mean()
gaussian_std = gaussian_accuracies.std()
non_gaussian_mean = non_gaussian_accuracies.mean()
non_gaussian_std = non_gaussian_accuracies.std()

print(f"\nOverall Statistics:")
print(f"Overall Mean Accuracy: {overall_mean:.2f}% (±{overall_std:.2f}%)")
print(f"Gaussian Mean Accuracy: {gaussian_mean:.2f}% (±{gaussian_std:.2f}%)")
print(f"Non-Gaussian Mean Accuracy: {non_gaussian_mean:.2f}% (±{non_gaussian_std:.2f}%)")

# Analyze cluster estimation
correct_estimations = (gaussian_data['n_centers'] == gaussian_data['estimated_n_centers']).mean()
print(f"\nCluster Estimation:")
print(f"Correct estimations: {correct_estimations:.2f}")