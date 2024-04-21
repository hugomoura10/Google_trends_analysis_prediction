import pandas as pd

# Load the dataset
data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)

# Convert 'Week' column to datetime format
data['Week'] = pd.to_datetime(data['Week'])

# Set 'Week' column as the index
data.set_index('Week', inplace=True)

# Resample the data to monthly frequency and aggregate using the mean (you can use 'sum' or other aggregation methods)
data_monthly = data.resample('M').mean()

# Reset the index to make 'Week' a column again
data_monthly.reset_index(inplace=True)

# Save the monthly data to a new CSV file
data_monthly.to_csv('Google Trends Data Challenge Datasets/personal_datasets/monthly_data.csv', index=False)