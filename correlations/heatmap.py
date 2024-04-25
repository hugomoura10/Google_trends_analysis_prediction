import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('correlation_results.csv', index_col=0)

cmap = sns.color_palette("coolwarm", as_cmap=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df, cmap=cmap, annot=True, fmt=".3f")
plt.title('Correlation between Token Price and Google Trends')
plt.xlabel('Week')
plt.ylabel('Token')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
