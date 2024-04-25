import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

token_names = ['bitcoin', 'ethereum', 'BNB', 'solana', 'XRP', 'dogecoin', 'cardano', 'polkadot', 'chainlink', 'litecoin', 'uniswap',
               'filecoin', 'fetch.ai', 'monero', 'singularitynet', 'tezos', 'kucoin', 'pancakeswap', 'oasis network', 'ocean protocol']

correlation_df = pd.DataFrame(index=token_names, columns=token_names)

for token1 in token_names:
    dataset1 = pd.read_csv(f'Google Trends Data Challenge Datasets/trends/{token1}.csv', skiprows=1)
    dataset1['Week'] = pd.to_datetime(dataset1['Week'])
    dataset1[f'{token1}: (Worldwide)'] = preprocess_trend_data(dataset1[f'{token1}: (Worldwide)']) 

    for token2 in token_names:
        if token1 == token2:
            correlation_df.loc[token1, token2] = 1.0
        else:
            dataset2 = pd.read_csv(f'Google Trends Data Challenge Datasets/trends/{token2}.csv', skiprows=1)
            dataset2['Week'] = pd.to_datetime(dataset2['Week'])
            dataset2[f'{token2}: (Worldwide)'] = preprocess_trend_data(dataset2[f'{token2}: (Worldwide)'])

            merged_data = pd.merge(dataset1, dataset2, on='Week')

            correlation = merged_data[f'{token1}: (Worldwide)'].corr(merged_data[f'{token2}: (Worldwide)'])
            correlation_df.loc[token1, token2] = correlation

correlation_df.to_csv('token_correlations.csv')

correlation_df = pd.read_csv('token_correlations.csv', index_col=0)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Token Trends')
plt.xlabel('Tokens')
plt.ylabel('Tokens')
plt.xticks(rotation=90)  
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
