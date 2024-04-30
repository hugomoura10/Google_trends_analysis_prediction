import pandas as pd

correlation_df = pd.read_csv('token_correlations.csv', index_col=0)
average_correlation = correlation_df.mean(axis=1)
token_ranking = average_correlation.sort_values(ascending=False).head(20)
print("Token Ranking (Top 20 Correlated Tokens by Average):")
print(token_ranking)
