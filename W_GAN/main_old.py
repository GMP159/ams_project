import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output
# Import the necessary function for discriminative score
from discriminative_score import discriminative_score_metric

# Define the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Tensor type based on the device
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Load the CSV data
data = pd.read_csv('cnc.csv')

# Select the relevant columns
data = data[['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 
              'materialremoved_sim', 'a_x', 'a_y', 'a_z', 'a_sp', 
              'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 
              'pos_z', 'pos_sp']]

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define the sequence length and number of features
seq_length = 50  # Example sequence length
num_features = data.shape[1]  # Number of features in the data

# Generate sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

sequences = create_sequences(data, seq_length)

# Convert to PyTorch tensors
train_data = torch.FloatTensor(sequences)

# Create a DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Hyperparameters
class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hp = Hyperparameters(
    n_epochs=1000,
    batch_size=batch_size,
    lr=0.0007,
    n_cpu=8,
    latent_dim=100,
    seq_length=seq_length,
    num_features=num_features,
    n_critic=5,
    clip_value=0.02,
    sample_interval=400,
)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, seq_length, latent_dim, num_features):
        super(Generator, self).__init__()

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
            nn.Tanh()  # Use Tanh for normalization
        )

    def forward(self, z):
        time_series = self.model(z)
        time_series = time_series.view(time_series.shape[0], hp.seq_length, hp.num_features)
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

# Initialize models
generator = Generator(hp.seq_length, hp.latent_dim, hp.num_features)
critic = Critic(hp.seq_length, hp.num_features)

# Move models to the selected device (GPU or CPU)
generator.to(device)
critic.to(device)

# Initialize optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=hp.lr)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=hp.lr)

# Weight initialization
def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

generator.apply(weights_init_normal)
critic.apply(weights_init_normal)

# Training function
def train():
    generated_data_list = []  # List to collect generated data
    for epoch in range(hp.n_epochs):
        for i, time_series in enumerate(train_loader):
            # Configure input
            real_time_series = Variable(time_series.type(Tensor)).to(device)

            # Train Critic
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (time_series.shape[0], hp.latent_dim)))).to(device)

            # Generate a batch of time series
            fake_time_series = generator(z).detach()

            # Loss for real and fake time series
            d_loss = -torch.mean(critic(real_time_series)) + torch.mean(critic(fake_time_series))
            d_loss.backward()

            # Clip gradients of the critic
            for param in critic.parameters():
                param.grad.data.clamp_(-hp.clip_value, hp.clip_value)

            optimizer_D.step()

            # Train the generator every n_critic iterations
            if i % hp.n_critic == 0:
                optimizer_G.zero_grad()

                # Generate a batch of time series
                gen_time_series = generator(z)

                # Loss for generator
                g_loss = -torch.mean(critic(gen_time_series))
                g_loss.backward()
                optimizer_G.step()

            # Log progress
            batches_done = epoch * len(train_loader) + i

            if batches_done % hp.sample_interval == 0:
                clear_output(wait=True)
                print(f"Epoch: {epoch} Batch: {i} D_loss: {d_loss.item()} G_loss: {g_loss.item()}")

                # Save the generated time series data
                if (epoch == 999 or epoch==998 or epoch==997 or epoch==996):
                    np.save(f"generated_timeseries_epoch_{epoch}_batch_{i}.npy", gen_time_series.cpu().data.numpy())
                elif (epoch == 200 or epoch==201 or epoch==202 or epoch==203):
                    np.save(f"generated_timeseries_epoch_{epoch}_batch_{i}.npy", gen_time_series.cpu().data.numpy())
                elif (epoch == 500 or epoch==501 or epoch==502 or epoch==503):
                    np.save(f"generated_timeseries_epoch_{epoch}_batch_{i}.npy", gen_time_series.cpu().data.numpy())

                # Store the generated data for later
                generated_data_list.append(gen_time_series.cpu().data.numpy())

                # Discriminative score calculation
                # Ensure real_data is a tensor before passing to the function
                real_data_tensor = torch.tensor(real_time_series.cpu().data.numpy(), device=device)  # Convert to tensor if necessary
                synthetic_data = gen_time_series.cpu().data.numpy()  # This is the synthetic data generated by the generator
                discriminative_score = discriminative_score_metric(
                    critic=critic, 
                    real_data=real_data_tensor,  # Use tensor version of real data
                    synthetic_data=synthetic_data, 
                    device=device  # Use the device defined earlier
                )

                print(f"Discriminative Score: {discriminative_score}")

    # Convert the collected generated data to a numpy array
    generated_data_array = np.concatenate(generated_data_list, axis=0)

    # Reshape for inverse transform
    generated_data_array = generated_data_array.reshape(-1, hp.seq_length * hp.num_features)

    # Inverse transform to get the original scale
    generated_data_array = scaler.inverse_transform(generated_data_array)

    # Save the generated data to a CSV file
    generated_df = pd.DataFrame(generated_data_array)
    generated_df.to_csv('synthetic_data.csv', index=False, header=False)

# Start the training
train()
