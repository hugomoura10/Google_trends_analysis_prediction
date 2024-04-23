import pandas as pd

def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

# Load the first dataset
dataset1 = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)
dataset1['Week'] = pd.to_datetime(dataset1['Week'])
dataset1['ethereum: (Worldwide)'] = preprocess_trend_data(dataset1['ethereum: (Worldwide)'])  # Replace 'bitcoin: (Worldwide)' with your column name

# Token names
token_names = ['bitcoin', 'BNB', 'solana', 'XRP', 'dogecoin', 'cardano', 'polkadot', 'chainlink', 'litecoin', 'uniswap',
               'filecoin', 'fetch.ai', 'monero', 'singularitynet', 'tezos', 'kucoin', 'pancakeswap', 'oasis network', 'ocean protocol']

# Initialize an empty dictionary to store correlations
correlation_values = {}

# Loop through each token name
for token_name in token_names:
    # Load the second dataset
    dataset2 = pd.read_csv(f'Google Trends Data Challenge Datasets/trends/{token_name.lower()}.csv', skiprows=1)
    dataset2['Week'] = pd.to_datetime(dataset2['Week'])
    dataset2[f'{token_name.lower()}: (Worldwide)'] = preprocess_trend_data(dataset2[f'{token_name}: (Worldwide)'])  # Adjust column name if necessary

    # Merge datasets on the 'Week' column
    merged_data = pd.merge(dataset1, dataset2, on='Week')

    # Compute correlation between the two datasets
    correlation = merged_data['ethereum: (Worldwide)'].corr(merged_data[f'{token_name.lower()}: (Worldwide)'])  # Adjust column name if necessary

    # Store correlation value in the dictionary
    correlation_values[token_name] = correlation

# Print correlation values
for token_name, correlation in correlation_values.items():
    print(f"Ethereum - {token_name}: {correlation}")
