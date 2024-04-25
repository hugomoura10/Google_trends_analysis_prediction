import pandas as pd

# Load correlation data into a DataFrame
correlation_df = pd.read_csv('token_correlations.csv', index_col=0)

# Calculate average correlation for each token
average_correlation = correlation_df.mean(axis=1)

# Sort tokens based on their average correlation
token_ranking = average_correlation.sort_values(ascending=False).head(20)

# Display the token ranking
print("Token Ranking (Top 20 Correlated Tokens by Average):")
print(token_ranking)
