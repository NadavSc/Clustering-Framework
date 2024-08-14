import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class GaussianMixture:
    def __init__(self, n_components, init_method='random', max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=42):
        self.log_likelihood_hist = []
        self.n_components = n_components
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        np.random.seed(self.random_state)

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        if self.init_method == 'random':
            self.weights = np.ones(self.n_components) / self.n_components
            self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
            self.covariances = [np.eye(n_features) for _ in range(self.n_components)]
        elif self.init_method == 'kmeans++':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_components, init='k-means++', n_init=1, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            self.weights = np.bincount(labels) / n_samples
            self.means = kmeans.cluster_centers_
            self.covariances = [np.cov(X[labels == k].T) + np.eye(n_features) * self.reg_covar for k in
                                range(self.n_components)]
        elif self.init_method == 'hcm':
            hcm = AgglomerativeClustering(n_clusters=self.n_components)
            labels = hcm.fit_predict(X)
            self.weights = np.bincount(labels) / n_samples
            self.means = np.array([X[labels == k].mean(axis=0) for k in range(self.n_components)])
            self.covariances = [np.cov(X[labels == k].T) + np.eye(n_features) * self.reg_covar for k in
                                range(self.n_components)]
        elif self.init_method == 'uniform':
            self.weights = np.ones(self.n_components) / self.n_components
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            self.means = np.random.uniform(min_vals, max_vals, size=(self.n_components, n_features))
            self.covariances = [np.eye(n_features) * np.mean((max_vals - min_vals) ** 2) / 12 for _ in
                                range(self.n_components)]
        elif self.init_method == 'stratified':
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.n_components, random_state=self.random_state)
            _, indices = next(sss.split(X, np.zeros(n_samples)))
            self.weights = np.ones(self.n_components) / self.n_components
            self.means = X[indices]
            self.covariances = [np.cov(X.T) + np.eye(n_features) * self.reg_covar for _ in range(self.n_components)]

    def fit(self, X):
        self.log_likelihood_hist = []
        self.initialize_parameters(X)

        for _ in range(self.max_iter):
            old_log_likelihood = self.log_likelihood(X)
            self.log_likelihood_hist.append(old_log_likelihood)

            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)

            new_log_likelihood = self.log_likelihood(X)
            if np.abs(new_log_likelihood - old_log_likelihood) < self.tol:
                break

    def e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        N = responsibilities.sum(axis=0)
        self.weights = N / X.shape[0]
        self.means = np.dot(responsibilities.T, X) / N[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N[k]
            self.covariances[k] += np.eye(X.shape[1]) * self.reg_covar

    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)

    def log_likelihood(self, X):
        likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihood[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        return np.sum(np.log(np.sum(likelihood, axis=1)))


class UOFC:
    def __init__(self, max_clusters=10, random_state=42, max_iter=100, m=2, error=1e-5):
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        np.random.seed(random_state)

    def initialize_membership(self, n_samples, n_clusters):
        return np.random.rand(n_samples, n_clusters)

    def update_centroids(self, X, membership):
        numerator = np.dot(X.T, np.power(membership, self.m))
        denominator = np.sum(np.power(membership, self.m), axis=0)
        return (numerator / denominator).T

    def update_membership(self, X, centroids):
        distances = cdist(X, centroids, metric='euclidean')
        return self.calculate_membership(distances)

    def calculate_membership(self, distances):
        power = 2 / (self.m - 1)
        tmp = np.power(1 / distances, power)
        denominator = np.sum(tmp, axis=1)
        return tmp / denominator[:, np.newaxis]

    def fcm(self, X, n_clusters):
        n_samples, n_features = X.shape
        membership = self.initialize_membership(n_samples, n_clusters)

        for _ in range(self.max_iter):
            old_membership = membership.copy()

            centroids = self.update_centroids(X, membership)
            membership = self.update_membership(X, centroids)

            if np.linalg.norm(membership - old_membership) < self.error:
                break

        return membership, centroids

    def fit(self, X):
        best_score = -np.inf
        best_n_clusters = 2
        best_membership = None
        best_centroids = None

        for n_clusters in range(2, self.max_clusters + 1):
            membership, centroids = self.fcm(X, n_clusters)

            labels = np.argmax(membership, axis=1)

            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_membership = membership
                    best_centroids = centroids

        self.n_clusters = best_n_clusters
        self.membership = best_membership
        self.centroids = best_centroids
        self.labels = np.argmax(best_membership, axis=1)
        return self.n_clusters, self.centroids, self.labels

    def predict(self, X):
        membership = self.update_membership(X, self.centroids)
        return np.argmax(membership, axis=1)


class HierarchicalClustering:
    def __init__(self, num_clusters, metric='max'):
        self.num_clusters = num_clusters
        self.metric = metric

    def fit(self, X):
        self.X = X
        self.n = X.shape[0]
        self.clusters = [[x] for x in X]
        self.current_num_clusters = self.n
        self.dendrogram = []

        while self.current_num_clusters > self.num_clusters:
            print(self.current_num_clusters)
            self.merge()

        self.labels = self.assign_cluster_labels(X)
        return self.labels, self.dendrogram

    def distance(self, cluster1, cluster2):
        if self.metric == 'min':
            return np.min([np.linalg.norm(x - y) for x in cluster1 for y in cluster2])
        elif self.metric == 'max':
            return np.max([np.linalg.norm(x - y) for x in cluster1 for y in cluster2])
        elif self.metric == 'avg':
            return np.mean([np.linalg.norm(x - y) for x in cluster1 for y in cluster2])
        elif self.metric == 'mean':
            centroid1 = np.mean(cluster1, axis=0)
            centroid2 = np.mean(cluster2, axis=0)
            return np.linalg.norm(centroid1 - centroid2)

    def merge(self):
        min_dist = float('inf')
        clusters_to_merge = (0, 0)

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                dist = self.distance(self.clusters[i], self.clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    clusters_to_merge = (i, j)

        i, j = clusters_to_merge
        self.clusters[i] = self.clusters[i] + self.clusters[j]
        self.clusters.pop(j)
        self.current_num_clusters -= 1
        self.dendrogram.append(((i, j), min_dist))

    def assign_cluster_labels(self, X):
        labels = []
        for point in X:
            for cluster_idx, cluster in enumerate(self.clusters):
                if any(np.array_equal(point, cluster_point) for cluster_point in cluster):
                    labels.append(cluster_idx)
                    break
        return labels
