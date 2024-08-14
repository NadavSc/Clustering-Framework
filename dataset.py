import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.datasets import make_blobs, load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def generate_clustered_rings(num_rings=3, points_per_ring=100, radius_increment=1, cluster_std=0.1):
    x, y = [], []
    for i in range(num_rings):
        theta = np.linspace(0, 2 * np.pi, points_per_ring)
        radius = radius_increment * (i + 1)
        ring_x = radius * np.cos(theta)
        ring_y = radius * np.sin(theta)
        x.extend(np.random.normal(ring_x, cluster_std))
        y.extend(np.random.normal(ring_y, cluster_std))
    return np.array(x), np.array(y)


def generate_wavy_pattern(waves=3, points_per_wave=100, wave_amplitude=1, wave_length=2 * np.pi, cluster_std=0.1):
    x, y = [], []
    for i in range(waves):
        wave_x = np.linspace(0, wave_length, points_per_wave)
        wave_y = wave_amplitude * np.sin(wave_x + i * (2 * np.pi / waves))
        x.extend(np.random.normal(wave_x, cluster_std))
        y.extend(np.random.normal(wave_y, cluster_std))
    return np.array(x), np.array(y)


def generate_concentric_circles(num_circles=3, points_per_circle=100, radius_increment=1):
    x, y = [], []
    for i in range(num_circles):
        theta = np.linspace(0, 2 * np.pi, points_per_circle)
        radius = radius_increment * (i + 1)
        x.extend(radius * np.cos(theta))
        y.extend(radius * np.sin(theta))
    return np.array(x), np.array(y)


def generate_multi_arm_spiral(arms=3, turns=2, points_per_turn=100, radius_increment=0.1):
    x, y = [], []
    for i in range(arms):
        theta = np.linspace(0, 2 * np.pi * turns, points_per_turn * turns)
        radius = radius_increment * theta
        angle_offset = i * (2 * np.pi / arms)
        x.extend((radius * np.cos(theta + angle_offset)))
        y.extend((radius * np.sin(theta + angle_offset)))
    return np.array(x), np.array(y)


def generate_gaussian_dataset(n_samples, n_features, n_centers,
                              center_box=(-10, 10), cluster_std=1.0,
                              random_state=None):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=n_centers, center_box=center_box,
                      cluster_std=cluster_std, random_state=random_state)
    return X, y


def load_real_world_dataset(dataset_name):
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError("Unknown dataset name")

    X, y = data.data, data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def plot_clusters(X, y, title):
    if X.shape[1] == 3:
        fig = plt.figure(figsize=(20, 8))

        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7)
        ax1.set_title(f"{title} (3D)")
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_zlabel('Feature 3')

        ax2 = fig.add_subplot(122)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax2.set_title(f"{title} (2D PCA)")
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')

    elif X.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    else:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title(f"{title} (2D PCA)")
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')

    plt.tight_layout()
    plt.show()


def plot_unique(shapes):
    plt.figure(figsize=(8, 8))
    for x, y, label in shapes:
        plt.scatter(x, y, label=label, alpha=0.6, edgecolor='k')
    plt.legend()
    plt.axis('equal')
    plt.show()


def generate_unique_data():
    dataset = {}

    x_rings, y_rings = generate_clustered_rings(num_rings=3, points_per_ring=100, radius_increment=1, cluster_std=0.1)
    x_wavy, y_wavy = generate_wavy_pattern(waves=3, points_per_wave=100, wave_amplitude=1, wave_length=2 * np.pi,
                                           cluster_std=0.1)
    x_circles, y_circles = generate_concentric_circles(num_circles=3, points_per_circle=100, radius_increment=1)
    x_spiral, y_spiral = generate_multi_arm_spiral(arms=3, turns=3, points_per_turn=75, radius_increment=0.1)

    # plot_unique([
    #     (x_rings, y_rings, 'Scattered Rings'),
    #     (x_wavy, y_wavy, 'DNA'),
    #     (x_circles, y_circles, 'Rings'),
    #     (x_spiral, y_spiral, 'Spiral')
    # ])

    dataset[0] = {'X': [(x, y) for x, y in zip(x_rings, y_rings)], 'type': 'scattered_ring'}
    dataset[1] = {'X': [(x, y) for x, y in zip(x_wavy, y_wavy)], 'type': 'dna'}
    dataset[2] = {'X': [(x, y) for x, y in zip(x_circles, y_circles)], 'type': 'rings'}
    dataset[3] = {'X': [(x, y) for x, y in zip(x_spiral, y_spiral)], 'type': 'spiral'}

    with open('data/dataset_unique.json', 'w') as f:
        json.dump(dataset, f)

    print('Dataset Unique has been saved')


def generate_data():
    dataset = {}
    random_state = 42

    idx = 0
    for n_features in range(2, 11):
        for n_centers in range(2, 8):
            for cluster_std in [0.5, 1.0, 2.0, 2.5]:
                X, y = generate_gaussian_dataset(1000, n_features, n_centers, cluster_std=cluster_std,
                                                 random_state=random_state)
                print(f"Simulation {idx + 1}")
                # plot_clusters(X, y, f'n_features: {n_features} | n_centers: {n_centers} | std: {cluster_std}')
                dataset[idx] = {'X': X.tolist(), 'y': y.tolist(), 'n_features': n_features, 'n_centers': n_centers,
                                'cluster_std': cluster_std, 'type': 'gaussian'}
                idx += 1

    for dataset_name in ['iris', 'wine', 'breast_cancer']:
        X, y = load_real_world_dataset(dataset_name)
        _, n_features = X.shape
        n_centers = len(np.unique(y))
        dataset[idx] = {'X': X.tolist(), 'y': y.tolist(), 'n_features': n_features, 'n_centers': n_centers, 'cluster_std': None,
                        'type': dataset_name}
        title = f"{dataset_name.capitalize()} Dataset"
        # plot_clusters(X, y, title)
        idx += 1

    with open('data/dataset.json', 'w') as f:
        json.dump(dataset, f)

    print('Dataset has been saved')


generate_data()
generate_unique_data()
