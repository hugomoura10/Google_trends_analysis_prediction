import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

# Load Google Trends data for the selected cryptocurrency token
data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)  
data.columns = ['ds', 'y']  
data['ds'] = pd.to_datetime(data['ds'])  

# Apply the Wiener filter to smooth the time series
smoothed_data = wiener(data['y'])

# Plot the original and smoothed time series
plt.plot(data['ds'], data['y'], label='Original', color='blue')
plt.plot(data['ds'], smoothed_data, label='Smoothed', color='red')
plt.xlabel('Date')
plt.ylabel('Search Interest')
plt.title('Original and Smoothed Time Series')
plt.legend()
plt.show()

# Prepare the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(smoothed_data.reshape(-1, 1))

# Define function to create dataset with lookback
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Split data into training and test sets
train_size = int(len(scaled_data) * 0.7)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

look_back = 12  # Adjust this parameter as needed
train_X, train_Y = create_dataset(train, look_back)
test_X, test_Y = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# Calculate mean absolute error
mae_train = mean_absolute_error(train_Y[0], train_predict[:,0])
mae_test = mean_absolute_error(test_Y[0], test_predict[:,0])

print(f"Mean Absolute Error on Training Data: {mae_train}")
print(f"Mean Absolute Error on Test Data: {mae_test}")
