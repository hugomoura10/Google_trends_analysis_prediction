import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

bitcoin_data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
bitcoin_data['Week'] = pd.to_datetime(bitcoin_data['Week'])
bitcoin_data.set_index('Week', inplace=True)
bitcoin_data.sort_index(inplace=True)

plt.plot(bitcoin_data.index, bitcoin_data['bitcoin: (Worldwide)'], label='Actual', color='g')
plt.title('Bitcoin Search Trends Over Time')
plt.xlabel('Week')
plt.ylabel('Search Trends')
plt.xticks(rotation=45)
plt.legend()
plt.show()

model = SARIMAX(bitcoin_data['bitcoin: (Worldwide)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()

last_fitted_values = result.fittedvalues.tail(3)
print("Last 3 Fitted Values (Predictions):\n", last_fitted_values)

last_actual_values = bitcoin_data['bitcoin: (Worldwide)'].tail(3)
print("\nLast 3 Actual Values:\n", last_actual_values)

future_steps = 262
future_index = pd.date_range(start=bitcoin_data.index[-1], periods=future_steps + 1, freq='W')[1:]
future = result.forecast(steps=future_steps)

plt.plot(bitcoin_data.index, bitcoin_data['bitcoin: (Worldwide)'], label='Actual', color='g')
plt.plot(result.fittedvalues.index, result.fittedvalues, label='Fitted', color='b')
plt.plot(future_index, future, label='Forecast', color='r')
plt.title('SARIMA Forecasting for Bitcoin Search Trends')
plt.xlabel('Week')
plt.ylabel('Search Trends')
plt.xticks(rotation=45)
plt.legend()
plt.show()

mae_historical = mean_absolute_error(bitcoin_data['bitcoin: (Worldwide)'], result.fittedvalues)
print(f"Mean Absolute Error on Historical Data: {mae_historical}")

if len(bitcoin_data['bitcoin: (Worldwide)']) == len(future):
    mae_future = mean_absolute_error(bitcoin_data['bitcoin: (Worldwide)'], future)
    print(f"Mean Absolute Error on Future Data: {mae_future}")
else:
    print("Lengths of actual and forecast data are inconsistent.")
