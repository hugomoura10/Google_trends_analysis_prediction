import pandas as pd

price_data = pd.read_csv('Google Trends Data Challenge Datasets/prices/BTC-USD.csv')
market_cap_data = pd.read_csv('Google Trends Data Challenge Datasets/additional_datasets/archive-6/bitcoin.csv')

price_data['Date'] = pd.to_datetime(price_data['Date'])
market_cap_data['date'] = pd.to_datetime(market_cap_data['date'])

merged_data = pd.merge(price_data, market_cap_data, left_on='Date', right_on='date', how='inner')

correlation = merged_data['Close'].corr(merged_data['market_cap'])

print(f"Correlation between Closing Price and Market Cap: {correlation}")