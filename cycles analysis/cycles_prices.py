import pandas as pd
import matplotlib.pyplot as plt

bitcoin_df = pd.read_csv('Google Trends Data Challenge Datasets/prices/BTC-USD.csv')

bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])

bitcoin_df = bitcoin_df[bitcoin_df['Date'] >= '2014-09-17']

cycle_1_start = '2014-09-17'
cycle_1_end = '2016-09-07'

cycle_2_start = '2016-09-07'
cycle_2_end = '2020-05-11'

cycle_1_data = bitcoin_df[(bitcoin_df['Date'] >= cycle_1_start) & (bitcoin_df['Date'] <= cycle_1_end)]
cycle_2_data = bitcoin_df[(bitcoin_df['Date'] >= cycle_2_start) & (bitcoin_df['Date'] <= cycle_2_end)]
cycle_3_data = bitcoin_df[bitcoin_df['Date'] >= cycle_2_end]

plt.figure(figsize=(10, 6))

# Plot cycle 1
plt.plot(cycle_1_data['Date'], cycle_1_data['Close'], label='Cycle 1')

# Plot cycle 2
plt.plot(cycle_2_data['Date'], cycle_2_data['Close'], label='Cycle 2')

# Plot cycle 3
plt.plot(cycle_3_data['Date'], cycle_3_data['Close'], label='Cycle 3')

plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, linewidth=0.2)
plt.show()

correlation = cycle_2_data['Close'].corr(cycle_3_data['Close'])

print("Correlation between the second and third cycles:", correlation)