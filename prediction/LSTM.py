import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Google Trends Data Challenge Datasets/trends/bitcoin.csv", skiprows=1)

# Preprocess the data
data['Week'] = pd.to_datetime(data['Week'])
data.set_index('Week', inplace=True)
data.sort_index(inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Convert data to PyTorch tensors
data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

# Define function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return torch.stack(sequences)

# Create sequences with a certain sequence length
seq_length = 5
sequences = create_sequences(data_tensor, seq_length)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
input_size = output_size = 1
hidden_size = 32
num_layers = 2
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for seq in sequences:
        optimizer.zero_grad()
        y_pred = model(seq[:-1].unsqueeze(0))
        loss = criterion(y_pred, seq[-1])
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
with torch.no_grad():
    future = 50
    preds = sequences[-1].unsqueeze(0)
    for _ in range(future):
        pred = model(preds[:, -seq_length:])
        preds = torch.cat([preds, pred.unsqueeze(0)], axis=1)

# Inverse scaling
preds = preds.squeeze().numpy()
preds = scaler.inverse_transform(preds.reshape(1, -1))

# Generate timestamps for predictions
pred_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=7), periods=future, freq='W')

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(data.index, data.values, label='Actual Data')

# Plot predictions for future dates
plt.plot(pred_dates, preds[0][-future:], label='Predictions', linestyle='--')

plt.xlabel('Week')
plt.ylabel('Bitcoin Trend')
plt.title('Bitcoin Trend Prediction')
plt.legend()
plt.show()
