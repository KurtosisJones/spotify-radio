import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class PCA():
    def __init__(self, components:int, X):
        self.components = components
        self.standarize_data = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

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

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 15),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def train(model, X, epochs, loss_function, optimizer):
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