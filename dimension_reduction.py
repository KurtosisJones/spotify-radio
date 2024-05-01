import numpy as np

class pca():
    def __init__(self, components:int):
        self.components = components

    def standarize_data(X):

        mean = np.mean(X, axis = 0)
        std_dev = np.sd(X, axis = 0)

        return (X - mean) / std_dev

    def get_covariance(X):
        return np.cov(X, rowvar=False)
    
    def get_eigenvalues(X_cov):
        eigenvalues, eigenvectors = np.linalg.eigh(X_cov)
        return eigenvalues[::-1], eigenvectors[:, ::-1]

    def get_pca(self, X):

        standard_X = self.standard_X(X)

        eigenvalues, eigenvectors = self.get_eigenvalues(self.get_covariance(X))

        pc = eigenvectors[:, :self.components]
        return standard_X @ pc
    
    def explained_variance(eigenvalues):
        total_variance = np.sum(eigenvalues)
        explained_variances = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variances)
        
        return explained_variances, cumulative_variance
