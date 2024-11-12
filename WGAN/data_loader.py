import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Function to load and preprocess the data
def load_data(file_path, seq_length):
    # Load the CSV data
    data = pd.read_csv(file_path)

    # Select the relevant columns
    data = data[['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 
                 'materialremoved_sim', 'a_x', 'a_y', 'a_z', 'a_sp', 
                 'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 
                 'pos_z', 'pos_sp']]

    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Create sequences
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    sequences = create_sequences(data, seq_length)
    train_data = torch.FloatTensor(sequences)

    return train_data, scaler

# Data loader function
def get_data_loader(train_data, batch_size):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader
