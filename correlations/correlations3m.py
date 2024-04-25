import pandas as pd
import os

def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

def calculate_correlation(token_symbol, trend_file_name, trend_column_name, max_weeks=10):
    price_csv_path = os.path.join('Google Trends Data Challenge Datasets', 'prices', f'{token_symbol.upper()}-USD.csv')
    trend_csv_path = os.path.join('Google Trends Data Challenge Datasets', 'trends', f'{trend_file_name}.csv')

    price_df = pd.read_csv(price_csv_path)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df.set_index('Date', inplace=True)
    
    trend_df = pd.read_csv(trend_csv_path, skiprows=1)
    trend_df['Week'] = pd.to_datetime(trend_df['Week'])
    trend_df.set_index('Week', inplace=True)
    trend_series = trend_df[trend_column_name]
    trend_series_numeric = preprocess_trend_data(trend_series)
    
    correlations = []
    for i in range(1, max_weeks + 1):
        
        price_df_resampled = price_df.resample(f'{i}W').mean()
        merged_df = pd.merge(price_df_resampled, trend_series_numeric, left_index=True, right_index=True)
        correlation = merged_df[trend_column_name].corr(merged_df['Close'])
        correlations.append(correlation)
        print(f"Correlation for {i} weeks: {correlation}")
    
    return correlations

def save_correlation_to_csv(token_list):
    df = pd.DataFrame(columns=[f'Week_{i}' for i in range(1, 11)])

    for token_symbol, trend_file_name, trend_column_name in token_list:
        correlations = calculate_correlation(token_symbol, trend_file_name, trend_column_name)
        df.loc[token_symbol] = correlations

    df.to_csv('correlation_results.csv')

token_list = [
    ('BTC', 'bitcoin', 'bitcoin: (Worldwide)'),
    ('ETH', 'ethereum', 'ethereum: (Worldwide)'),
    ('BNB', 'bnb', 'BNB: (Worldwide)'),
    ('ADA', 'cardano', 'cardano: (Worldwide)'),
    ('LINK', 'chainlink', 'chainlink: (Worldwide)'),
    ('DOGE', 'dogecoin', 'dogecoin: (Worldwide)'),
    ('FET', 'fetch.ai', 'fetch.ai: (Worldwide)'),
    ('FIL', 'filecoin', 'filecoin: (Worldwide)'),
    ('KCS', 'kucoin', 'kucoin: (Worldwide)'),
    ('LTC', 'litecoin', 'litecoin: (Worldwide)'),
    ('XMR', 'monero', 'monero: (Worldwide)'),
    ('ROSE', 'oasis network', 'oasis network: (Worldwide)'),
    ('OCEAN', 'ocean protocol', 'ocean protocol: (Worldwide)'),
    ('CAKE', 'pancakeswap', 'pancakeswap: (Worldwide)'),
    ('DOT', 'polkadot', 'polkadot: (Worldwide)'),
    ('AGIX', 'singularitynet', 'singularitynet: (Worldwide)'),
    ('SOL', 'solana', 'solana: (Worldwide)'),
    ('XTZ', 'tezos', 'tezos: (Worldwide)'),
    ('UNI', 'uniswap', 'uniswap: (Worldwide)'),
    ('XRP', 'xrp', 'XRP: (Worldwide)'),
]

save_correlation_to_csv(token_list)
