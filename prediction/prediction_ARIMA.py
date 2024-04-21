import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from neuralprophet import NeuralProphet
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

# Create a new DataFrame with the smoothed data
smoothed_df = pd.DataFrame({'ds': data['ds'], 'y': smoothed_data})

# Initialize and fit the NeuralProphet model with the smoothed data
model = NeuralProphet()
model.fit(smoothed_df)

# Make future predictions
future = model.make_future_dataframe(smoothed_df, periods=262) # 61 months for future prediction
forecast = model.predict(future)
actual_prediction = model.predict(smoothed_df)

plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label="prediction_Actual", c='r')
plt.plot(forecast['ds'], forecast['yhat1'], label='future_prediction', c='b')
plt.plot(smoothed_df['ds'], smoothed_df['y'], label='actual', c='g')
plt.legend()
plt.title('Prediction')
plt.show()

model.plot_components(forecast)

# Calculate mean absolute error on historical data
mae_historical = mean_absolute_error(actual_prediction['y'], actual_prediction['yhat1'])
print(f"Mean Absolute Error on Historical Data: {mae_historical}")

# Calculate mean absolute error on future data
mae_future = mean_absolute_error(smoothed_df['y'], forecast['yhat1'].iloc[:-61])  # Exclude the historical part
print(f"Mean Absolute Error on Future Data: {mae_future}")