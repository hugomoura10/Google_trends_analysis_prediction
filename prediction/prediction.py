import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import wiener
import matplotlib.pyplot as plt

# Load merged dataset with both search trend and price data
df = pd.read_csv('bitcoin_merged_data.csv')
df = df.dropna()
df['Week'] = pd.to_datetime(df['Week'])

# Smoothing the bitcoin search trend using the Wiener filter
df['bitcoin_smoothed'] = wiener(df['bitcoin: (Worldwide)'], mysize=3)

# Features and target variable
X = df[['bitcoin_smoothed', 'Close']][-3:]  # Features (last three rows)
y = df['bitcoin: (Worldwide)'][-3:]         # Target (last three rows)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X, y)

# Prediction for the last three rows
predictions = model.predict(X)
print('The model predicts the last three rows:', predictions)
print('Actual values:', y)
