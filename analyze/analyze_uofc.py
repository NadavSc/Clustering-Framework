import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('results/results_uofc.csv')

# Print column names to verify
print("Columns in the dataset:", data.columns.tolist())


# Function to plot accuracy comparison
def plot_accuracy_comparison(df, x, y, hue, title):
    if x not in df.columns or y not in df.columns or hue not in df.columns:
        print(f"Error: One or more columns ({x}, {y}, {hue}) not found in the dataframe.")
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, hue=hue, data=df)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.show()


# Function to plot heatmap
def plot_heatmap(df, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# 1. Overall accuracy comparison
plot_accuracy_comparison(data, 'criterion', 'accuracy', 'cluster_std',
                         'Accuracy Comparison Across Criteria and Cluster Std')

# 2. Accuracy comparison for different number of centers
plot_accuracy_comparison(data[data['n_centers'].isin([2, 3, 4])], 'criterion', 'accuracy', 'n_centers',
                         'Accuracy Comparison Across Criteria and Number of Centers')

# 3. Heatmap of average accuracy for each criterion and cluster_std
heatmap_data = data.pivot_table(values='accuracy', index='criterion', columns='cluster_std', aggfunc='mean')
plot_heatmap(heatmap_data, 'Average Accuracy Heatmap: Criterion vs Cluster Std')

# 4. Analyze the impact of cluster_std on accuracy
std_impact = data.groupby('cluster_std')['accuracy'].mean().sort_values(ascending=False)
print("Impact of cluster_std on average accuracy:")
print(std_impact)

# 5. Analyze the performance of different criteria
criteria_performance = data.groupby('criterion')['accuracy'].mean().sort_values(ascending=False)
print("\nPerformance of different criteria:")
print(criteria_performance)

# 6. Analyze the impact of number of centers on accuracy
centers_impact = data.groupby('n_centers')['accuracy'].mean().sort_values(ascending=False)
print("\nImpact of number of centers on average accuracy:")
print(centers_impact)

# 7. Analyze the relationship between estimated_n_centers and actual n_centers
data['center_estimation_error'] = abs(data['estimated_n_centers'] - data['n_centers'])
center_estimation = data.groupby('criterion')['center_estimation_error'].mean().sort_values()
print("\nAverage center estimation error for each criterion:")
print(center_estimation)

# 8. Plot the relationship between cluster_std and accuracy for each criterion
plt.figure(figsize=(12, 6))
for criterion in data['criterion'].unique():
    criterion_data = data[data['criterion'] == criterion]
    plt.plot(criterion_data['cluster_std'], criterion_data['accuracy'], label=criterion, marker='o')
plt.xlabel('Cluster Std')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Cluster Std for Each Criterion')
plt.legend()
plt.grid(True)
plt.show()

# Print overall insights
print("\nKey Insights:")
print("1. The most accurate criteria across all conditions are:", ", ".join(criteria_performance.head(3).index))
print("2. The least accurate criteria are:", ", ".join(criteria_performance.tail(3).index))
print(f"3. Cluster standard deviation has the biggest impact on accuracy at {std_impact.index[0]}")
print(f"4. The number of centers with the highest average accuracy is {centers_impact.index[0]}")
print(f"5. The criterion with the lowest center estimation error is {center_estimation.index[0]}")