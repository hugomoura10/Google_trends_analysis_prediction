import pandas as pd

# Read the daily dataset
daily_df = pd.read_csv('Google Trends Data Challenge Datasets/prices/BTC-USD.csv')

# Convert the 'Date' column to datetime
daily_df['Date'] = pd.to_datetime(daily_df['Date'])

# Set the 'Date' column as the index
daily_df.set_index('Date', inplace=True)

# Resample the dataset to weekly frequency and take the mean of each week
weekly_df = daily_df.resample('W').mean()

# Optionally, you can interpolate missing values to fill in any gaps
weekly_df = weekly_df.interpolate(method='linear')

# Reset the index to make 'Date' a regular column again
weekly_df = weekly_df.reset_index()

# Save the weekly dataset to a new CSV file
weekly_df.to_csv('weekly_dataset_btc.csv', index=False)