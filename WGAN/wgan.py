import torch
import torch.nn as nn
import numpy as np

# Define the Generator
class Generator(nn.Module):
    def __init__(self, seq_length, latent_dim, num_features):
        super(Generator, self).__init__()
        self.seq_length = seq_length  # Set seq_length as a class attribute
        self.num_features = num_features  # Set num_features as a class attribute

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, seq_length * num_features),
            nn.Tanh()
        )

    def forward(self, z):
        time_series = self.model(z)
        # Use self.seq_length and self.num_features here
        time_series = time_series.view(time_series.shape[0], self.seq_length, self.num_features)
        return time_series


# Define the Critic
class Critic(nn.Module):
    def __init__(self, seq_length, num_features):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(seq_length * num_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, time_series):
        time_series_flat = time_series.view(time_series.shape[0], -1)
        validity = self.model(time_series_flat)
        return validity

# Function to initialize weights
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# Hyperparameters container
class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
