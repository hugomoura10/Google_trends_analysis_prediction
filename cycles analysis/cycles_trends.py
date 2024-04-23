import pandas as pd
import matplotlib.pyplot as plt

# Read your trend dataset CSV file
trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)

# Convert 'Week' column to datetime format
trend_df['Week'] = pd.to_datetime(trend_df['Week'])

# Define the date ranges for each cycle
cycle_1_start = '2019-04-07'
cycle_1_end = '2020-05-10'

cycle_2_start = '2020-05-10'

# Filter data for each cycle
cycle_1_data = trend_df[(trend_df['Week'] >= cycle_1_start) & (trend_df['Week'] <= cycle_1_end)]
cycle_2_data = trend_df[trend_df['Week'] >= cycle_2_start]

# Plot the two cycles on the same plot
plt.figure(figsize=(10, 6))
# Plot cycle 1 in green
plt.plot(cycle_1_data['Week'], cycle_1_data['ethereum: (Worldwide)'], color='orange', label='Cycle 2')

# Plot cycle 2 in orange
plt.plot(cycle_2_data['Week'], cycle_2_data['ethereum: (Worldwide)'], color='green', label='Cycle 3')

plt.title('ethereum Search Trends')
plt.xlabel('Week')
plt.ylabel('Search Volume')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True,linewidth=0.2)

# Show the plot with both cycles
plt.show()
