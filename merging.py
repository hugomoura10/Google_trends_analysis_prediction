import pandas as pd

search_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)
search_trend_df['Week'] = pd.to_datetime(search_trend_df['Week'])

price_df = pd.read_csv('Google Trends Data Challenge Datasets/prices/ETH-USD.csv')
price_df['Date'] = pd.to_datetime(price_df['Date'])

merged_df = pd.merge(search_trend_df, price_df, how='inner', left_on='Week', right_on='Date')

merged_df.to_csv('ethereum_merged_data.csv', index=False)