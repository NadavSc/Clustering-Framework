import json
import os
os.environ['OMP_NUM_THREADS'] = "2"

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
matplotlib.use('TkAgg')

from models import GaussianMixture, UOFC, HierarchicalClustering


def clustering_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # Hungarian algorithm to find the optimal assignment

    accuracy = cm[row_ind, col_ind].sum() / cm.sum()
    return accuracy*100


def plot_clusters(X, true_labels, predicted_labels, title, accuracy, example_info):
    if X.shape[1] > 3:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(example_info)
    if X.shape[1] == 3:
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=true_labels, cmap='viridis', alpha=0.7)
        ax1.set_title(f"{' '.join(title.split(' ')[:2])} (3D)\nTrue Clusters")
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_zlabel('Feature 3')

        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=predicted_labels, cmap='viridis', alpha=0.7)
        ax2.set_title(f"{title}\nPredicted Clusters\nAccuracy: {accuracy:.2f}")
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.set_zlabel('Feature 3')
        plt.tight_layout()
        plt.show()
    else:
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=true_labels, cmap='viridis')
        ax1.set_title(f"{' '.join(title.split(' ')[:2])}\nTrue Clusters")
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')

        ax2 = fig.add_subplot(122)
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=predicted_labels, cmap='viridis')
        ax2.set_title(f"{title}\nPredicted Clusters\nAccuracy: {accuracy:.2f}")
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()


def run_uofc(results, dataset_type, max_clusters, n_centers, random_state, X, y, idx,
             n_features, cluster_std, criteria, verbose):
    for criterion in criteria:
        uofc = UOFC(max_clusters=max_clusters, random_state=random_state, criterion=criterion)
        estimated_n_centers, centers, y_pred = uofc.fit(X)
        accuracy = clustering_accuracy(y, y_pred)

        if verbose > 1:
            example_info = f'n_features: {n_features} | n_centers: {n_centers} | std: {cluster_std}'
            plot_clusters(X, y, y_pred, f"Simulation {idx + 1} - UOFC", accuracy, example_info)
        results.append({
            'id': idx,
            'algorithm': 'uofc',
            'dataset_type': dataset_type,
            'n_features': n_features,
            'n_centers': n_centers,
            'estimated_n_centers': estimated_n_centers,
            'cluster_std': cluster_std,
            'init_method': None,
            'distance_metric': None,
            'criterion': criterion,
            'accuracy': accuracy
        })
    return results


def run_gmm(results, dataset_type, init_methods, n_centers, random_state, X, y, idx, n_features, cluster_std, verbose):
    for init_method in init_methods:
        gmm = GaussianMixture(n_components=n_centers, init_method=init_method,
                              random_state=random_state)
        gmm.fit(X)
        y_pred = gmm.predict(X)
        accuracy = clustering_accuracy(y, y_pred)
        if verbose > 1:
            example_info = f'n_features: {n_features} | n_centers: {n_centers} | std: {cluster_std}'
            plot_clusters(X, y, y_pred, f"Example {idx + 1} - GMM-{init_method}", accuracy, example_info)

        results.append({
            'id': idx,
            'algorithm': 'gmm',
            'dataset_type': dataset_type,
            'n_features': n_features,
            'n_centers': n_centers,
            'estimated_n_centers': None,
            'cluster_std': cluster_std,
            'init_method': init_method,
            'distance_metric': None,
            'criterion': None,
            'accuracy': accuracy
        })
    return results


def run_hc(results, dataset_type, n_centers, X, y, idx, n_features, cluster_std, distance_metrics, verbose):
    for distance_metric in distance_metrics:
        hc = HierarchicalClustering(num_clusters=n_centers, metric=distance_metric)
        hc.fit(X)

        y_pred = hc.assign_cluster_labels(X)
        accuracy = clustering_accuracy(y, y_pred)
        if verbose > 1:
            example_info = f'n_features: {n_features} | n_centers: {n_centers} ' \
                           f'| std: {cluster_std} | distance-metric: {distance_metric}'
            plot_clusters(X, y, y_pred, f"Simulation {idx + 1}", accuracy, example_info)

        results.append({
            'id': idx,
            'algorithm': 'hc',
            'dataset_type': dataset_type,
            'n_features': n_features,
            'n_centers': n_centers,
            'estimated_n_centers': None,
            'cluster_std': cluster_std,
            'init_method': None,
            'distance_metric': distance_metric,
            'criterion': None,
            'accuracy': accuracy
        })
    return results


def run_algorithms(dataset, algorithms, verbose):
    results = []
    random_state = 42
    max_clusters = 7
    init_methods = ['random', 'kmeans++', 'hcm', 'uniform', 'stratified']
    distance_metrics = ['min', 'max', 'avg', 'mean']
    uofc_criteria = ['silhouette', 'trace', 'hypervolume', 'partition_density',
                     'avg_partition_density', 'max_avg_partition_density',
                     'normalized_partition_coeff', 'invariant']

    for idx in range(len(dataset)):
        json_object = dataset[str(idx)]
        X, y = np.array(json_object['X']), np.array(json_object['y'])
        n_features = json_object['n_features']
        n_centers = json_object['n_centers']
        cluster_std = json_object['cluster_std']
        dataset_type = json_object['type']

        if 'uofc' in algorithms:
            results = run_uofc(results, dataset_type, max_clusters, n_centers, random_state, X, y, idx,
                               n_features, cluster_std, uofc_criteria, verbose)
        if 'gmm' in algorithms:
            results = run_gmm(results, dataset_type, init_methods, n_centers, random_state, X, y, idx,
                              n_features, cluster_std, verbose)
        if 'hc' in algorithms:
            results = run_hc(results, dataset_type, n_centers, X, y, idx, n_features, cluster_std,
                             distance_metrics, verbose)

        print(f'Example {idx} has been processed')
    results = pd.DataFrame(results)
    results.to_csv('results/results.csv', index=False)


if __name__ == "__main__":
    verbose = 0
    algorithms = ['gmm', 'uofc', 'hc']
    with open('data/dataset.json', 'r') as openfile:
        dataset = json.load(openfile)

    run_algorithms(dataset, algorithms, verbose)
