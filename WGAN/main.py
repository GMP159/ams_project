import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
from data_loader import load_data, get_data_loader
from wgan import Generator, Critic, Hyperparameters, weights_init_normal
from metrics.discriminative_score import discriminative_score_metric
from IPython.display import clear_output

# Define device and Tensor type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Load data
seq_length = 50  # Example sequence length
file_path = 'data/cnc.csv'
train_data, scaler = load_data(file_path, seq_length)

# Create DataLoader
batch_size = 64
train_loader = get_data_loader(train_data, batch_size)

# Set up hyperparameters
hp = Hyperparameters(
    n_epochs=1000,
    batch_size=batch_size,
    lr=0.0007,
    n_cpu=8,
    latent_dim=100,
    seq_length=seq_length,
    num_features=train_data.shape[2],  # Number of features in the data
    n_critic=5,
    clip_value=0.02,
    sample_interval=400,
)

# Initialize models
generator = Generator(hp.seq_length, hp.latent_dim, hp.num_features).to(device)
critic = Critic(hp.seq_length, hp.num_features).to(device)
generator.apply(weights_init_normal)
critic.apply(weights_init_normal)

# Initialize optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=hp.lr)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=hp.lr)

# Training function
def train():
    generated_data_list = []  # List to collect generated data
    for epoch in range(hp.n_epochs):
        for i, time_series in enumerate(train_loader):
            real_time_series = Variable(time_series.type(Tensor)).to(device)

            # Train Critic
            optimizer_D.zero_grad()
            z = torch.tensor(np.random.normal(0, 1, (time_series.shape[0], hp.latent_dim)), dtype=torch.float, device=device)
            fake_time_series = generator(z).detach()
            d_loss = -torch.mean(critic(real_time_series)) + torch.mean(critic(fake_time_series))
            d_loss.backward()

            # Clip gradients of the critic
            for param in critic.parameters():
                param.grad.data.clamp_(-hp.clip_value, hp.clip_value)
            optimizer_D.step()

            # Train Generator
            if i % hp.n_critic == 0:
                optimizer_G.zero_grad()
                gen_time_series = generator(z)
                g_loss = -torch.mean(critic(gen_time_series))
                g_loss.backward()
                optimizer_G.step()

            # Log progress
            batches_done = epoch * len(train_loader) + i
            if batches_done % hp.sample_interval == 0:
                clear_output(wait=True)
                print(f"Epoch: {epoch} Batch: {i} D_loss: {d_loss.item()} G_loss: {g_loss.item()}")

                # Save generated data periodically
                generated_data_list.append(gen_time_series.cpu().data.numpy())
                discriminative_score = discriminative_score_metric(
                    critic=critic, 
                    real_data=real_time_series.cpu().data.numpy(),
                    synthetic_data=gen_time_series.cpu().data.numpy(),
                    device=device
                )
                print(f"Discriminative Score: {discriminative_score}")

    # Process and save final generated data
    generated_data_array = np.concatenate(generated_data_list, axis=0)
    generated_data_array = generated_data_array.reshape(-1, hp.seq_length * hp.num_features)
    generated_data_array = scaler.inverse_transform(generated_data_array)
    generated_df = pd.DataFrame(generated_data_array)
    generated_df.to_csv('synthetic_data.csv', index=False, header=False)

# Start training
train()
