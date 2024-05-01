import unittest
import numpy as np
import torch
import os 
import sys
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lambda_payload.helper.dimension_reduction import PCA, AutoEncoder

class TestPCA(unittest.TestCase):
    def setUp(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        self.pca = PCA(components=2, X=X)

    def test_standardization(self):
        np.testing.assert_allclose(np.mean(self.pca.standarize_data, axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(np.std(self.pca.standarize_data, axis=0), 1, atol=1e-7)

    def test_covariance_matrix(self):
        cov_matrix = self.pca.get_covariance()
        self.assertEqual(cov_matrix.shape, (3, 3))
        self.assertTrue((np.diag(cov_matrix) >= 0).all())

    def test_eigen_decomposition(self):
        eigenvalues, eigenvectors = self.pca.get_eigenvalues()
        self.assertTrue(np.all(np.diff(eigenvalues) <= 0))
        self.assertEqual(eigenvectors.shape, (3, 3))

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        X = np.random.randn(100, 20)
        self.model = AutoEncoder(latent_components=10, transport_layer_count=2, X=X)
        self.sample_data = torch.randn(10, 20)

    def test_encoder_decoder_existence(self):
        self.assertIsInstance(self.model.encoder, nn.Sequential)
        self.assertIsInstance(self.model.decoder, nn.Sequential)

    def test_forward_pass(self):
        output = self.model.forward(self.sample_data)
        self.assertEqual(output.shape, self.sample_data.shape)

    def test_encoder_output(self):
        encoded_output = self.model.get_encoder(self.sample_data)
        self.assertEqual(encoded_output.shape, (10, self.model.latent_components))

    def test_decoder_output(self):
        encoded_output = self.model.get_encoder(self.sample_data)
        decoded_output = self.model.get_decoder(encoded_output)
        self.assertEqual(decoded_output.shape, self.sample_data.shape)

if __name__ == '__main__':
    unittest.main()