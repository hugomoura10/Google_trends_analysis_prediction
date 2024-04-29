import pandas as pd
import matplotlib.pyplot as plt
import os

def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

token_names = ['bitcoin', 'ethereum', 'BNB', 'solana', 'XRP', 'dogecoin', 'cardano', 'polkadot', 'chainlink', 'litecoin', 'uniswap',
               'filecoin', 'fetch.ai', 'monero', 'singularitynet', 'tezos', 'kucoin', 'pancakeswap', 'oasis network', 'ocean protocol']


additional_dataset_path = "Google Trends Data Challenge Datasets/additional_datasets/crypto_searches_weekly.csv"
additional_df = pd.read_csv(additional_dataset_path, skiprows=1)

additional_df['Week'] = pd.to_datetime(additional_df['Week'])
additional_df.set_index('Week', inplace=True)
correlation_results = {}

for token in token_names:
    file_path = os.path.join("Google Trends Data Challenge Datasets/trends", f"{token}.csv")  # Change this to the path where your CSV files are stored
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, skiprows=1)
        
        df['Week'] = pd.to_datetime(df['Week'])
        df.set_index('Week', inplace=True)
        
        correlation_results[token] = preprocess_trend_data(df[f"{token}: (Worldwide)"])
    else:
        print(f"File not found for {token}")

concatenated_data = pd.concat([additional_df] + list(correlation_results.values()), axis=1, join='inner')
correlation_values = concatenated_data.corr().iloc[0, 1:]
correlation_values_sorted = correlation_values.sort_values(ascending=False)
correlation_values_sorted.index = [label.split(':')[0] for label in correlation_values_sorted.index]

plt.figure(figsize=(12, 8))
correlation_values_sorted.plot(kind='bar', color='skyblue')
plt.title('Correlation between Crypto Searches and Tokens')
plt.xlabel('Tokens')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
