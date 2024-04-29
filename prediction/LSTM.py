import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

 
bitcoin_data = pd.read_csv("Google Trends Data Challenge Datasets/trends/bitcoin.csv", skiprows=1)

 
bitcoin_data['Week'] = pd.to_datetime(bitcoin_data['Week'])
bitcoin_data.set_index('Week', inplace=True)
bitcoin_data.sort_index(inplace=True)

 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_bitcoin_data = scaler.fit_transform(bitcoin_data)

 
bitcoin_data_tensor = torch.tensor(scaled_bitcoin_data, dtype=torch.float32)

 
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return torch.stack(sequences)

 
sequence_length = 5
bitcoin_sequences = create_sequences(bitcoin_data_tensor, sequence_length)

 
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

 
input_size = output_size = 1
hidden_size = 128
num_layers = 3
bitcoin_model = LSTMModel(input_size, hidden_size, num_layers, output_size)

 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(bitcoin_model.parameters(), lr=0.0001)

 
num_epochs = 50
for epoch in range(num_epochs):
    for seq in bitcoin_sequences:
        optimizer.zero_grad()
        y_pred = bitcoin_model(seq[:-1].unsqueeze(0))
        loss = criterion(y_pred, seq[-1])
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

 
with torch.no_grad():
    future = 50
    bitcoin_preds = bitcoin_sequences[-1].unsqueeze(0)
    for _ in range(future):
        pred = bitcoin_model(bitcoin_preds[:, -sequence_length:])
        bitcoin_preds = torch.cat([bitcoin_preds, pred.unsqueeze(0)], axis=1)

 
bitcoin_preds = bitcoin_preds.squeeze().numpy()
bitcoin_preds = scaler.inverse_transform(bitcoin_preds.reshape(1, -1)).flatten()

 
future_dates = pd.date_range(start=bitcoin_data.index[-1] + pd.Timedelta(days=7), periods=future, freq='W')

 
plt.figure(figsize=(10, 5))
plt.plot(bitcoin_data.index, bitcoin_data.values, label='Actual Data')

 
plt.plot(future_dates, bitcoin_preds[-future:], label='Predictions', linestyle='--')

plt.xlabel('Week')
plt.ylabel('Bitcoin Trend')
plt.title('Bitcoin Trend Prediction')
plt.legend()
plt.show()

bitcoin_preds = bitcoin_preds.squeeze().numpy()
bitcoin_preds = scaler.inverse_transform(bitcoin_preds.reshape(1, -1)).flatten()

future_dates = pd.date_range(start=bitcoin_data.index[-1] + pd.Timedelta(days=7), periods=bitcoin_preds.shape[0], freq='W')

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
