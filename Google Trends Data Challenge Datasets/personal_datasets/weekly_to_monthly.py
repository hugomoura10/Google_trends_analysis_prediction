import pandas as pd

data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
data['Week'] = pd.to_datetime(data['Week'])
data.set_index('Week', inplace=True)
data_monthly = data.resample('M').mean()
data_monthly.reset_index(inplace=True)
data_monthly.to_csv('Google Trends Data Challenge Datasets/personal_datasets/monthly_data.csv', index=False)