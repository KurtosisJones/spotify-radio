import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

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
        
        # Adjust the input size to match the dataset features
        self.encoder = nn.Sequential(
            nn.Linear(15, 5),  # Adjusted from 10 to 15
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 15),  # Adjusted from 10 to 15
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def train(model, dataloader, epochs, loss_function, optimizer):
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
    
    def compress_data(encoder, dataloader):
        compressed_data = []
        encoder.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Turn off gradients for this block
            for batch in dataloader:
                inputs = batch[0]
                compressed_output = encoder(inputs)
                compressed_data.append(compressed_output)

        # Concatenate all batches
        compressed_data = torch.cat(compressed_data, dim=0)
        return compressed_data