import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the data
df = pd.read_csv('../results/results_hc.csv')

# Convert accuracy to range 0-1
df['accuracy'] = df['accuracy'] / 100

# Separate Gaussian and non-Gaussian datasets
gaussian_df = df[df['dataset_type'] == 'gaussian']
non_gaussian_df = df[df['dataset_type'] != 'gaussian']


# Function to plot accuracy vs. cluster_std for each distance metric and n_centers
def plot_accuracy_vs_cluster_std(data):
    n_centers_values = sorted(data['n_centers'].unique())
    metrics = data['distance_metric'].unique()
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(metrics)))

    fig, axes = plt.subplots(len(n_centers_values), 1, figsize=(12, 6 * len(n_centers_values)), sharex=True)
    fig.suptitle('Accuracy vs. Cluster Std Dev for Gaussian Datasets', fontsize=16)

    for idx, n_centers in enumerate(n_centers_values):
        ax = axes[idx] if len(n_centers_values) > 1 else axes
        subset = data[data['n_centers'] == n_centers]
        for metric, color in zip(metrics, colors):
            metric_data = subset[subset['distance_metric'] == metric]
            ax.plot(metric_data['cluster_std'], metric_data['accuracy'], marker='o', label=metric, color=color)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'n_centers = {n_centers}')
        ax.legend()
        ax.grid(True)

    plt.xlabel('Cluster Standard Deviation')
    plt.tight_layout()
    plt.show()


# Plot for Gaussian datasets
plot_accuracy_vs_cluster_std(gaussian_df)


# Function to calculate average accuracy for each distance metric
def avg_accuracy_by_metric(data):
    return data.groupby('distance_metric')['accuracy'].mean().sort_values(ascending=False)


# Calculate and print average accuracies
print("Average Accuracy by Distance Metric (Gaussian):")
print(avg_accuracy_by_metric(gaussian_df))
print("\nAverage Accuracy by Distance Metric (Non-Gaussian):")
print(avg_accuracy_by_metric(non_gaussian_df))


# Function to perform t-test between two distance metrics
def perform_ttest(data, metric1, metric2):
    group1 = data[data['distance_metric'] == metric1]['accuracy']
    group2 = data[data['distance_metric'] == metric2]['accuracy']
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value


# Perform t-tests for Gaussian datasets
distance_metrics = gaussian_df['distance_metric'].unique()
print("\nT-test results for Gaussian datasets:")
for i in range(len(distance_metrics)):
    for j in range(i + 1, len(distance_metrics)):
        t_stat, p_value = perform_ttest(gaussian_df, distance_metrics[i], distance_metrics[j])
        print(f"{distance_metrics[i]} vs {distance_metrics[j]}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# Plot barplot for non-Gaussian datasets
plt.figure(figsize=(12, 6))
custom_palette = sns.color_palette("Set2")
sns.barplot(x='dataset_type', y='accuracy', hue='distance_metric', data=non_gaussian_df, ci='sd', palette=custom_palette, edgecolor='black')
plt.title('Accuracy Distribution for Non-Gaussian Datasets')
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.show()


# Function to create heatmap for performance by number of features and centers
def heatmap_by_features_and_centers(data, title):
    pivot = data.pivot_table(values='accuracy', index=['n_features'],
                             columns=['n_centers', 'distance_metric'], aggfunc='mean')
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1)
    plt.title(title)
    plt.show()


print("\nPerformance Heatmap by Features and Centers (Gaussian):")
heatmap_by_features_and_centers(gaussian_df, "Performance Heatmap for Gaussian Datasets")

print("\nPerformance Heatmap by Features and Centers (Non-Gaussian):")
heatmap_by_features_and_centers(non_gaussian_df, "Performance Heatmap for Non-Gaussian Datasets")
