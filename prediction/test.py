import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load data
df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
df = df.dropna()

# Convert 'Week' column to datetime
df['Week'] = pd.to_datetime(df['Week'])

# Plot original data
df.plot(x="Week", y="bitcoin: (Worldwide)")
plt.xticks(rotation=45)
plt.title('Original bitcoin Searches')
plt.show()

# Apply Wiener smoothing to the data
smoothed_data = wiener(df['bitcoin: (Worldwide)'])

# Update DataFrame with smoothed data
df['bitcoin_smoothed'] = smoothed_data

# Plot smoothed data
df.plot(x="Week", y="bitcoin_smoothed")
plt.xticks(rotation=45)
plt.title('Smoothed bitcoin Searches')
plt.show()

# Prepare data for modeling
X = df[['bitcoin_smoothed']][:-3]  # Features (excluding the last 3 rows)
y = df['bitcoin_smoothed'][3:]     # Target (excluding the first 3 rows)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Model evaluation
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print('Train Score:', train_score)
print('Test Score:', test_score)

# Prediction for the last 3 rows
new_data = df[['bitcoin_smoothed']].tail(3)
predictions = model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(df['bitcoin_smoothed'].tail(3))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(df.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(df.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('bitcoin Searches Forecast (Smoothed)')
plt.xlabel('Week')
plt.ylabel('bitcoin Searches (Smoothed)')
plt.legend()
plt.show()
