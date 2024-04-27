import numpy as np
from numpy.linalg import norm

class Cluster:
    '''Initialize generic cluster options'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialize_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids
    
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

class Kmeans(Cluster):
    '''Implementing Kmeans algorithm.'''

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)

class Kmedians(Cluster):

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.median(X[labels == k, :], axis=0)
        return centroids
    
    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(centroids):
            distance[:, k] = norm(X - centroids[k, :], axis=1, ord=1)
        return distance

    def update_medians(self, X, labels):
        new_medians = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_medians[k] = np.median(cluster_points, axis=0)
        return new_medians

    def compute_sad(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1, ord=1)
        return np.sum(distance)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distances = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distances)
            self.centroids = self.update_medians(X, self.labels)
            if np.array_equal(centroids, self.centroids):
                break
            centroids = self.centroids
        self.error = self.compute_sse(X, self.labels, self.centroids)

class DBScnan:

    def __init__(self, tol:float = 0.001, minimal_points:int = 10, random_state:int = 123):
        self.tol = tol
        self.minimal_points = minimal_points
        self.random_state = random_state

    def initialize_clusters(X):

        visited, noise = [False] * X.shape[0]

        return np.column_stack((X, visited, noise))

    def query(self, p, X):
        distance = np.sum(np.square(norm(X - X[:,p], axis = 1)))
        return distance < self.tol

    def add_to_cluster(self, p, nn, k, X_prime):
        for i in range(len(nn)):
            if nn[i]:
                X_prime[i, -1] = k


    def scan(self, X):
        X_prime = self.initialize_clusters(X)
        k = 0
        for p in range(X_prime.shape[0]):
            if not X_prime[p, -2]:
                X_prime[p, -2] = True
                nn = self.query(p, X)
                if np.sum(nn) < self.minimal_points:
                    X_prime[p, -1] = True
                else:
                    k += 1
                    self.add_to_cluster(p, nn, k, X_prime)
        return X_prime

    

        