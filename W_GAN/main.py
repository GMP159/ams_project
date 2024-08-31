import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

plt.ion()
from IPython.display import clear_output

# Load the CSV data
data = pd.read_csv('data.csv')


# Select the relevant columns (assuming all columns are relevant)
data = data[['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 
             'materialremoved_sim', 'a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 
             'pos_y', 'pos_z', 'pos_sp']]

# Convert DataFrame to numpy array
data = data.to_numpy()

# Define the sequence length and number of features
seq_length = 50  # Example sequence length
num_features = data.shape[1]  # Number of features in the data

# Generate sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

sequences = create_sequences(data, seq_length)

# Convert to PyTorch tensors
train_data = torch.FloatTensor(sequences)

# Create a DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hp = Hyperparameters(
    n_epochs=1000,
    batch_size=64,
    lr=0.00001,
    n_cpu=8,
    latent_dim=100,
    seq_length=seq_length,
    num_features=num_features,
    n_critic=5,
    clip_value=0.005,
    sample_interval=400,
)

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
            nn.Tanh()
        )

        self.seq_length = seq_length
        self.num_features = num_features

    def forward(self, z):
        time_series = self.model(z)
        time_series = time_series.view(time_series.shape[0], self.seq_length, self.num_features)
        return time_series

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

        self.seq_length = seq_length
        self.num_features = num_features

    def forward(self, time_series):
        time_series_flat = time_series.view(time_series.shape[0], -1)
        validity = self.model(time_series_flat)
        return validity

# Initialize models
generator = Generator(hp.seq_length, hp.latent_dim, hp.num_features)
critic = Critic(hp.seq_length, hp.num_features)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator.cuda()
    critic.cuda()

# Initialize optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=hp.lr)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=hp.lr)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

generator.apply(weights_init_normal)
critic.apply(weights_init_normal)

# Initialize optimizers with reduced learning rate
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=0.00005)

def train():
    for epoch in range(hp.n_epochs):
        for i, time_series in enumerate(train_loader):

            # Configure input
            real_time_series = Variable(time_series.type(Tensor))

            # Train Critic
            optimizer_D.zero_grad()
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (time_series.shape[0], hp.latent_dim))))
            
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
                print(f"Epoch:{epoch} Batch:{i} D_loss:{d_loss.item()} G_loss:{g_loss.item()}")
                
                # Save the generated time series data
                if (epoch == 999 or epoch==998 or epoch==997 or epoch==996):
                    np.save(f"generated_timeseries_epoch_{epoch}_batch_{i}.npy", gen_time_series.cpu().data.numpy())
                elif (epoch == 200 or epoch==201 or epoch==202 or epoch==203):
                    np.save(f"generated_timeseries_epoch_{epoch}_batch_{i}.npy", gen_time_series.cpu().data.numpy())
                elif (epoch == 500 or epoch==501 or epoch==502 or epoch==503):
                    np.save(f"generated_timeseries_epoch_{epoch}_batch_{i}.npy", gen_time_series.cpu().data.numpy())


# Start the training
train()
