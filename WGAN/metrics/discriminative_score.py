# discriminative_score.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def discriminative_score_metric(critic, real_data, synthetic_data, device, test_size=0.3, batch_size=32, num_epochs=100):
    """
    Calculate the discriminative score for WGAN.

    Args:
    - critic: Pre-initialized WGAN Critic model
    - real_data: Real data samples
    - synthetic_data: Generated synthetic data samples
    - device: Device to run the model on ('cpu' or 'cuda')
    - test_size: Fraction of the dataset to be used as test data
    - batch_size: Batch size for training
    - num_epochs: Number of epochs to train the discriminator

    Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
    """

    # Combine real and synthetic data
    real_data = real_data.cpu().detach().numpy()  # Remove GPU dependency
    combined_data = np.vstack([real_data, synthetic_data])  # Combine real and generated data
    labels = np.array([1]*real_data.shape[0] + [0]*synthetic_data.shape[0])  # 1 for real, 0 for synthetic

    # Train-test split
    train_x, test_x, train_y, test_y = train_test_split(combined_data, labels, test_size=test_size, random_state=42)

    # Convert to torch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    # Training the Critic
    criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy with logits loss
    optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        critic.train()

        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            optimizer.zero_grad()

            # Forward pass through the Critic
            predictions = critic(batch_x.unsqueeze(1)).squeeze(1)  # Assuming batch_x shape is [batch_size, seq_length, num_features]
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation of the Critic on test data
    critic.eval()
    with torch.no_grad():
        predictions = critic(test_x.unsqueeze(1)).squeeze(1)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        accuracy = accuracy_score(test_y.cpu().numpy(), predictions.cpu().numpy())

    # Discriminative score: difference from 0.5 (perfect accuracy)
    discriminative_score = np.abs(0.5 - accuracy)

    return discriminative_score
