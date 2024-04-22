import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import yfinance as yf
from sklearn.metrics import mean_absolute_error

# Load Google Trends data for the selected cryptocurrency token
data = pd.read_csv('Google Trends Data Challenge Datasets/personal_datasets/monthly_data.csv')  
data.columns = ['ds', 'y']  
data['ds'] = pd.to_datetime(data['ds'])  

# Plot the actual Google Trends data
plt.plot(data['ds'], data['y'], label='Actual', color='g')
plt.xlabel('Date')
plt.ylabel('Search Interest')
plt.title('Google Trends Data Over Time')
plt.legend()
plt.show()

# Initialize and fit the NeuralProphet model
model = NeuralProphet()
model.fit(data)

# Make future predictions
future = model.make_future_dataframe(data, periods=61) #262 for week
forecast = model.predict(future)
actual_prediction = model.predict(data)

plt.plot(actual_prediction['ds'], actual_prediction['yhat1'], label = "prediction_Actual", c = 'r')
plt.plot(forecast['ds'], forecast['yhat1'], label = 'future_prediction', c = 'b')
plt.plot(data['ds'], data['y'], label = 'actual', c = 'g')
plt.legend()
plt.title('Prediction')
plt.show()

model.plot_components(forecast)
mae_historical = mean_absolute_error(actual_prediction['y'], actual_prediction['yhat1'])
print(f"Mean Absolute Error on Historical Data: {mae_historical}")

# Calculate mean absolute error on future data if applicable
if len(data) == len(forecast):
    mae_future = mean_absolute_error(data['y'], forecast['yhat1'])
    print(f"Mean Absolute Error on Future Data: {mae_future}")
else:
    print("Lengths of actual and forecast data are inconsistent.")
