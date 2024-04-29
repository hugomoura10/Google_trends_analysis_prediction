import pandas as pd
import numpy as np

def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

def calculate_correlation(price_csv_path, trend_csv_path, trend_column_name):
    price_df = pd.read_csv(price_csv_path)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df.set_index('Date', inplace=True)
    
    trend_df = pd.read_csv(trend_csv_path, skiprows=1)
    trend_df['Week'] = pd.to_datetime(trend_df['Week'])
    trend_df.set_index('Week', inplace=True)
    trend_series = trend_df[trend_column_name]
    trend_series_numeric = preprocess_trend_data(trend_series)
    
    
    price_df_resampled = price_df.resample('W').mean()
    
    merged_df = pd.merge(price_df_resampled, trend_series_numeric, left_index=True, right_index=True)
    print(merged_df)
    correlation = merged_df[trend_column_name].corr(merged_df['Close'])
    
    return correlation

price_csv_path = 'Google Trends Data Challenge Datasets/prices/FIL-USD.csv'
trend_csv_path = 'Google Trends Data Challenge Datasets/trends/filecoin.csv'
trend_column_name = 'filecoin: (Worldwide)' 
correlation = calculate_correlation(price_csv_path, trend_csv_path, trend_column_name)
print(f"Correlation between '{trend_column_name}' trend and price: {correlation}")