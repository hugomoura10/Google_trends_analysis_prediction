import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

 
data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)

 
data['Week'] = pd.to_datetime(data['Week'])
data.set_index('Week', inplace=True)

 
smoothed_data = wiener(data['bitcoin: (Worldwide)'])

 
plt.plot(data.index, data['bitcoin: (Worldwide)'], label='Original', color='grey', linewidth=0.9)
plt.plot(data.index, smoothed_data, label='Smoothed', color='blue', linewidth=0.7)
plt.xlabel('Week')
plt.ylabel('Bitcoin Trend')
plt.title('Wiener Filter Smoothing')
plt.legend()
plt.show()