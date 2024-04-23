import pandas as pd

# Define preprocessing function
def preprocess_trend_data(trend_series):
    # Replace '<1' with 0
    trend_series_numeric = trend_series.replace('<1', 0)
    # Convert to numeric values
    trend_series_numeric = pd.to_numeric(trend_series_numeric, errors='coerce').fillna(0)
    return trend_series_numeric

# Load market cap data
market_cap_data = pd.read_csv('Google Trends Data Challenge Datasets/additional_datasets/archive-6/bitcoin_modified.csv')

# Load trend data
trend_data = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)

# Convert 'date' columns to datetime format
market_cap_data['date'] = pd.to_datetime(market_cap_data['date'])
trend_data['Week'] = pd.to_datetime(trend_data['Week'])

# Preprocess trend data
trend_data['bitcoin: (Worldwide)'] = preprocess_trend_data(trend_data['bitcoin: (Worldwide)'])

# Merge datasets on the 'date' and 'Week' columns
merged_data = pd.merge(market_cap_data, trend_data, left_on='date', right_on='Week')

# Calculate correlation between 'market_cap' and 'bitcoin: (Worldwide)' columns
correlation = merged_data['market_cap'].corr(merged_data['bitcoin: (Worldwide)'])

print(merged_data)
print(f"Correlation between Market Cap and bitcoin Trend: {correlation}")
