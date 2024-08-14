import json
import os
os.environ['OMP_NUM_THREADS'] = "2"

import numpy as np

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from models import GaussianMixture, UOFC, HierarchicalClustering


def clustering_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # Hungarian algorithm to find the optimal assignment

    accuracy = cm[row_ind, col_ind].sum() / cm.sum()
    return accuracy*100


def plot_clusters(X, predicted_labels, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
    plt.title(f"{title}\nPredicted Clusters")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.show()


def run_uofc(X, dataset_type):
    uofc = UOFC(max_clusters=max_clusters, random_state=random_state)
    estimated_n_centers, centers, y_pred = uofc.fit(X)

    plot_clusters(X, y_pred, f"UOFC - Simulation {idx + 1} - {dataset_type}")


def run_gmm(X, dataset_type, n_clusters, init_methods):
    for init_method in init_methods:
        gmm = GaussianMixture(n_components=n_clusters, init_method=init_method,
                              random_state=random_state)
        gmm.fit(X)
        y_pred = gmm.predict(X)
        plot_clusters(X, y_pred, f"GMM: {init_method} | Simulation {idx + 1} - {dataset_type}")


def run_hc(X, dataset_type, n_clusters, distance_metrics):
    for distance_metric in distance_metrics:
        hc = HierarchicalClustering(num_clusters=n_clusters, metric=distance_metric)
        hc.fit(X)

        y_pred = hc.assign_cluster_labels(X)
        plot_clusters(X, y_pred, f"HC Simulation {idx + 1} - {dataset_type}")


if __name__ == "__main__":
    algorithms = ['hc', 'gmm', 'uofc']

    random_state = 42
    max_clusters = 7
    init_methods = ['random', 'kmeans++', 'hcm', 'uniform', 'stratified']
    distance_metrics = ['min', 'max', 'avg', 'mean']

    with open('data/dataset_unique.json', 'r') as openfile:
        # Reading from json file
        dataset = json.load(openfile)

    for idx in range(len(dataset)):
        json_object = dataset[str(idx)]
        X = np.array(json_object['X'])
        dataset_type = json_object['type']
        n_clusters = 3

        if 'uofc' in algorithms:
            run_uofc(X, dataset_type)

        if 'hc' in algorithms:
            run_hc(X, dataset_type, n_clusters, distance_metrics)

        if 'gmm' in algorithms:
            run_gmm(X, dataset_type, n_clusters, init_methods)
