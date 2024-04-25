import pandas as pd

daily_df = pd.read_csv('Google Trends Data Challenge Datasets/prices/BTC-USD.csv')

daily_df['Date'] = pd.to_datetime(daily_df['Date'])

daily_df.set_index('Date', inplace=True)

weekly_df = daily_df.resample('W').mean()

weekly_df = weekly_df.interpolate(method='linear')

weekly_df = weekly_df.reset_index()

weekly_df.to_csv('weekly_dataset_btc.csv', index=False)