import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class PCA():
    def __init__(self, components:int, X):
        self.components = components
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.standarize_data = (X - self.mean) / self.std
        self.eigenvalues, self.eigenvectors = self.get_eigenvalues()
        self.pc = self.eigenvectors[:, :self.components]

    def get_covariance(self):
        return np.cov(self.standarize_data, rowvar=False)
    
    def get_eigenvalues(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.get_covariance())
        return eigenvalues[::-1], eigenvectors[:, ::-1]

    def get_pca(self):

        eigenvalues, eigenvectors = self.get_eigenvalues()

        pc = eigenvectors[:, :self.components]
        return self.standarize_data @ pc
    
    def explained_variance(self):
        total_variance = np.sum(self.get_eigenvalues()[0])
        explained_variances = self.get_eigenvalues()[0] / total_variance
        cumulative_variance = np.cumsum(explained_variances)

        return explained_variances, cumulative_variance
    
    def project(self, X_new):
        X_new_standardized = (X_new - self.mean) / self.std
        return X_new_standardized @ self.pc


class AutoEncoder(nn.Module):
    def __init__(self, latent_space):
        super(AutoEncoder, self).__init__()
        self.latent_space = latent_space

        if self.latent_space >= 5:
            raise Exception("latentent space cannot be the same dimension or larger than intermediates")
        
        self.encoder = nn.Sequential(
            nn.Linear(13, 5),
            nn.ReLU(),
            nn.Linear(5, self.latent_space),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space, 5),
            nn.ReLU(),
            nn.Linear(5, 13),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def custom_train(model, X, epochs, loss_function, optimizer):
        standarize_data = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        tensor_data = torch.tensor(standarize_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                inputs = batch[0]
                outputs = model(inputs)
                loss = loss_function(outputs, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    def compress_data(self, X):
        standarize_data = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        tensor_data = torch.tensor(standarize_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        compressed_data = []
        self.encoder.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0]
                compressed_output = self.encoder(inputs)
                compressed_data.append(compressed_output)

        compressed_data = torch.cat(compressed_data, dim=0)
        return compressed_data