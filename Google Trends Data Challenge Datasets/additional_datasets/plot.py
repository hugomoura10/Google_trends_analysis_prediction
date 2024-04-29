import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Google Trends Data Challenge Datasets/additional_datasets/crypto_searches.csv', skiprows=1)

df['Month'] = pd.to_datetime(df['Month'])

plt.figure(figsize=(8, 4)) 
plt.plot(df['Month'], df['crypto: (Worlwide)'], linestyle='-', linewidth=0.7)
plt.xlabel('Month')
plt.ylabel('Searches')
plt.grid(True, linewidth=0.2)
plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()
