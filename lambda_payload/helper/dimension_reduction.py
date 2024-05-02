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

    def __init__(self, latent_components:int, transport_layer_count:int, X):
        super(AutoEncoder, self).__init__()
        self.latent_components = latent_components
        self.transport_layer_count = transport_layer_count
        self.features = torch.from_numpy(X).float()

        self.encoder = nn.Sequential(
            nn.Linear(self.__len__(), 8),
            nn.ReLU(),
            nn.Linear(8, latent_components),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_components, 8),
            nn.ReLU(),
            nn.Linear(8, self.__len__()),
            nn.ReLU()
        )

    def __len__(self):
        return self.features.shape[1]
    
    # not implimented will be at some point
    def transition_layers(self):

        jumps = int((self.__len__() - self.latent_components) / self.transport_layer_count)

        new_layers = []
        current_size = self.__len__()

        for i in range(1, self.transport_layer_count + 1):
            next_size = current_size - jumps
            new_layers.append(nn.Linear(current_size, next_size))
            new_layers.append(nn.ReLU())
            current_size = next_size

        return(new_layers)

    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, epochs, learning_rate):
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            print(f"start... epoch: {epoch}")
            for inputs in self.features:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"end... epoch: {epoch} --- loss: {total_loss} --- Total time: {time.time() - start_time:.2f}")
    
    def get_encoder(self, data):
        self.eval()

        with torch.no_grad():
            predictions = self.encoder(data)
        return predictions
    
    def get_decoder(self, data):
        self.eval()

        with torch.no_grad():
            predictions = self.decoder(data)
        return predictions