import pandas as pd
import matplotlib.pyplot as plt

def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

market_cap_data = pd.read_csv('Google Trends Data Challenge Datasets/additional_datasets/archive-6/bitcoin_modified.csv')

trend_data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)

market_cap_data['date'] = pd.to_datetime(market_cap_data['date'])
trend_data['Week'] = pd.to_datetime(trend_data['Week'])

trend_data['bitcoin: (Worldwide)'] = preprocess_trend_data(trend_data['bitcoin: (Worldwide)'])

merged_data = pd.merge(market_cap_data, trend_data, left_on='date', right_on='Week')

correlation = merged_data['market_cap'].corr(merged_data['bitcoin: (Worldwide)'])

print(merged_data)
print(f"Correlation between Market Cap and bitcoin Trend: {correlation}")

correlation_values = {
    'Bitcoin': 0.641,
    'Ethereum': 0.662,
    'Cardano': 0.847,
    'Chainlink': 0.630,
    'Filecoin': 0.533,
    'Polkadot': 0.799,
    'Litecoin': 0.751,
    'Monero': 0.719,
    'BNB': 0.286,
    'XRP': 0.588,
    'Uniswap': 0.767,
    'Solana': 0.866,
    'Dogecoin': 0.546
}

sorted_correlation_values = dict(sorted(correlation_values.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(12, 6))
plt.bar(sorted_correlation_values.keys(), sorted_correlation_values.values(), color='skyblue')
plt.title('Correlation between Monthly Crypto and Weekly Token Trends')
plt.xlabel('Token')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()