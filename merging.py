import pandas as pd

# Load search trend data
search_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
search_trend_df['Week'] = pd.to_datetime(search_trend_df['Week'])

# Load price data
price_df = pd.read_csv('Google Trends Data Challenge Datasets/prices/BTC-USD.csv')
price_df['Date'] = pd.to_datetime(price_df['Date'])

# Merge datasets based on common timestamps
merged_df = pd.merge(search_trend_df, price_df, how='inner', left_on='Week', right_on='Date')

# Save merged data to CSV file
merged_df.to_csv('bitcoin_merged_data.csv', index=False)