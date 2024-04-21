import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load the weekly dataset
data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)

# Convert 'Week' column to datetime format and set it as index
data['Week'] = pd.to_datetime(data['Week'])
data.set_index('Week', inplace=True)

# Apply the Wiener filter to smooth the time series
smoothed_data = wiener(data['bitcoin: (Worldwide)'])

# Plot the original and smoothed time series
plt.plot(data.index, data['bitcoin: (Worldwide)'], label='Original', color='blue')
plt.plot(data.index, smoothed_data, label='Smoothed', color='red')
plt.xlabel('Week')
plt.ylabel('Bitcoin Trend')
plt.title('Wiener Filter Smoothing')
plt.legend()
plt.show()