import pandas as pd

# Load market cap data
market_cap_data = pd.read_csv('Google Trends Data Challenge Datasets/additional_datasets/archive-6/ethereum_modified.csv')

# Load trend data
trend_data = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)

# Convert 'date' columns to datetime format
market_cap_data['date'] = pd.to_datetime(market_cap_data['date'])
trend_data['Week'] = pd.to_datetime(trend_data['Week'])

# Merge datasets on the 'date' and 'Week' columns
merged_data = pd.merge(market_cap_data, trend_data, left_on='date', right_on='Week')

# Calculate correlation between 'market_cap' and 'bitcoin: (Worldwide)' columns
correlation = merged_data['market_cap'].corr(merged_data['ethereum: (Worldwide)'])

print(merged_data)

print(f"Correlation between Market Cap and Ethereum Trend: {correlation}")