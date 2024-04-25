import pandas as pd
import matplotlib.pyplot as plt

trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)

trend_df['Week'] = pd.to_datetime(trend_df['Week'])

cycle_1_start = '2019-04-07'
cycle_1_end = '2020-05-10'

cycle_2_start = '2020-05-10'

cycle_1_data = trend_df[(trend_df['Week'] >= cycle_1_start) & (trend_df['Week'] <= cycle_1_end)]
cycle_2_data = trend_df[trend_df['Week'] >= cycle_2_start]

plt.figure(figsize=(10, 6))
plt.plot(cycle_1_data['Week'], cycle_1_data['ethereum: (Worldwide)'], color='orange', label='Cycle 2')
plt.plot(cycle_2_data['Week'], cycle_2_data['ethereum: (Worldwide)'], color='green', label='Cycle 3')

plt.title('ethereum Search Trends')
plt.xlabel('Week')
plt.ylabel('Search Volume')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True,linewidth=0.2)

plt.show()
