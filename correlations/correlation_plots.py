import pandas as pd 
import matplotlib.pyplot as plt

correlation_data = pd.read_csv('Google Trends Data Challenge Datasets/personal_datasets/correlations_values.csv')

correlation_data_sorted = correlation_data.sort_values(by='correlation', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(correlation_data_sorted['Token'], correlation_data_sorted['correlation'], color='skyblue')
plt.title('Correlation between Tokens and Price')
plt.xlabel('Token')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

