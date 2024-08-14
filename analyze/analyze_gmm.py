import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_dataset_type(data):
    dataset_types = data['dataset_type'].unique()
    multiple_types = len(dataset_types) > 1

    if multiple_types:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset_type', y='accuracy', data=data)
        plt.title('Accuracy by Dataset Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Dataset Type')
        plt.tight_layout()
        plt.show()

        type_avg = data.groupby('dataset_type')['accuracy'].mean().sort_values(ascending=False)
        print("Average accuracy by dataset type:")
        print(type_avg)
        custom_palette = sns.color_palette("Set2")

        plt.figure(figsize=(14, 6))
        sns.barplot(x='init_method', y='accuracy', hue='dataset_type', data=data, palette=custom_palette, edgecolor='black')
        plt.title('Accuracy by Initialization Method and Dataset Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Initialization Method')
        plt.legend(title='Dataset Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        type_init_avg = data.groupby(['dataset_type', 'init_method'])['accuracy'].mean().unstack()
        print("\nAverage accuracy by dataset type and initialization method:")
        print(type_init_avg)

        # Analyze performance difference between Gaussian and non-Gaussian datasets
        data['accuracy_diff'] = data.groupby('init_method')['accuracy'].transform(lambda x: x.iloc[0] - x.iloc[1])
        plt.figure(figsize=(12, 6))
        sns.barplot(x='init_method', y='accuracy_diff', data=data.drop_duplicates('init_method'))
        plt.title('Performance Difference: Gaussian - Non-Gaussian')
        plt.ylabel('Accuracy Difference')
        plt.xlabel('Initialization Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='init_method', y='accuracy', data=data)
    plt.title('Accuracy by Initialization Method')
    plt.ylabel('Accuracy')
    plt.xlabel('Initialization Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    init_method_avg = data.groupby('init_method')['accuracy'].mean().sort_values(ascending=False)
    print("\nAverage accuracy by initialization method:")
    print(init_method_avg)

    conditions = ['n_centers', 'cluster_std']
    for condition in conditions:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=condition, y='accuracy', hue='init_method', data=data, marker='o')
        plt.title(f'Initialization Method Performance by {condition}')
        plt.ylabel('Accuracy')
        plt.xlabel(condition)
        plt.legend(title='Init Method')
        plt.tight_layout()
        plt.show()

    data['config'] = data['n_centers'].astype(str) + '_' + data['cluster_std'].astype(str)
    best_init = data.loc[data.groupby('config')['accuracy'].idxmax()]
    init_method_counts = best_init['init_method'].value_counts()
    print("\nCount of configurations where each init method performs best:")
    print(init_method_counts)

    worst_case = data.groupby('init_method')['accuracy'].min().sort_values()
    print("\nWorst-case accuracy for each initialization method:")
    print(worst_case)

    init_method_std = data.groupby('init_method')['accuracy'].std().sort_values()
    print("\nStandard deviation of accuracy for each initialization method:")
    print(init_method_std)

    difficult_scenarios = data[(data['cluster_std'] >= 2) | (data['n_centers'] >= 5)]
    difficult_avg = difficult_scenarios.groupby('init_method')['accuracy'].mean().sort_values(ascending=False)
    print("\nAverage accuracy in difficult scenarios by initialization method:")
    print(difficult_avg)


def analyze_init_method(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='init_method', y='accuracy', data=data)
    plt.title('Accuracy by Initialization Method')
    plt.ylabel('Accuracy')
    plt.xlabel('Initialization Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    init_method_avg = data.groupby('init_method')['accuracy'].mean().sort_values(ascending=False)
    print("Average accuracy by initialization method:")
    print(init_method_avg)

    conditions = ['n_centers', 'cluster_std']
    for condition in conditions:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=condition, y='accuracy', hue='init_method', data=data, marker='o')
        plt.title(f'Initialization Method Performance by {condition}')
        plt.ylabel('Accuracy')
        plt.xlabel(condition)
        plt.legend(title='Init Method')
        plt.tight_layout()
        plt.show()

    data['config'] = data['n_centers'].astype(str) + '_' + data['cluster_std'].astype(str)
    best_init = data.loc[data.groupby('config')['accuracy'].idxmax()]
    init_method_counts = best_init['init_method'].value_counts()
    print("\nCount of configurations where each init method performs best:")
    print(init_method_counts)

    worst_case = data.groupby('init_method')['accuracy'].min().sort_values()
    print("\nWorst-case accuracy for each initialization method:")
    print(worst_case)

    init_method_std = data.groupby('init_method')['accuracy'].std().sort_values()
    print("\nStandard deviation of accuracy for each initialization method:")
    print(init_method_std)

    difficult_scenarios = data[(data['cluster_std'] >= 2) | (data['n_centers'] >= 5)]
    difficult_avg = difficult_scenarios.groupby('init_method')['accuracy'].mean().sort_values(ascending=False)
    print("\nAverage accuracy in difficult scenarios by initialization method:")
    print(difficult_avg)


results_df = pd.read_csv('../results/results_gmm.csv')
analyze_init_method(results_df)
