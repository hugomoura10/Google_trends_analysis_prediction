import pandas as pd
import matplotlib.pyplot as plt

correlation_data = pd.read_csv('Google Trends Data Challenge Datasets/personal_datasets/correlations_trends_btc.csv')

correlation_data_sorted = correlation_data.sort_values(by='value', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(correlation_data_sorted['tokens'], correlation_data_sorted['value'], color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Tokens')
plt.ylabel('Correlation')
plt.title('Correlation between Bitcoin and Various Tokens')
plt.tight_layout()
plt.show()
