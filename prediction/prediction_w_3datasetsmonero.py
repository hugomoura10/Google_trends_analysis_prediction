import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Function to apply Wiener smoothing
def apply_smoothing(data):
    return wiener(data)

# Load Bitcoin search trend data
bitcoin_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
bitcoin_trend_df = bitcoin_trend_df.dropna()
bitcoin_trend_df['Week'] = pd.to_datetime(bitcoin_trend_df['Week'])

# Load Ethereum search trend data
ethereum_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)
ethereum_trend_df['Week'] = pd.to_datetime(ethereum_trend_df['Week'])

# Load Monero search trend data
monero_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/monero.csv', skiprows=1)
monero_trend_df['Week'] = pd.to_datetime(monero_trend_df['Week'])

# Merge Bitcoin, Ethereum, and Monero search trend data
merged_data = pd.merge(bitcoin_trend_df, ethereum_trend_df, on='Week', how='left')
merged_data = pd.merge(merged_data, monero_trend_df, on='Week', how='left')

# Choose whether to apply smoothing or not
apply_smoothing_flag = True  # Set this to False if you don't want to apply smoothing
if apply_smoothing_flag:
    # Apply Wiener smoothing to the bitcoin search trend data
    merged_data['bitcoin_smoothed'] = apply_smoothing(merged_data['bitcoin: (Worldwide)'])
    merged_data['ethereum_smoothed'] = apply_smoothing(merged_data['ethereum: (Worldwide)'])
    merged_data['monero_smoothed'] = apply_smoothing(merged_data['monero: (Worldwide)'])
else:
    merged_data['bitcoin_smoothed'] = merged_data['bitcoin: (Worldwide)']
    merged_data['ethereum_smoothed'] = merged_data['ethereum: (Worldwide)']
    merged_data['monero_smoothed'] = merged_data['monero: (Worldwide)']

# Prepare data for modeling
X = merged_data[['bitcoin_smoothed', 'ethereum_smoothed', 'monero_smoothed']][:-3]  # Features (excluding the last 3 rows)
y = merged_data['bitcoin_smoothed'][3:]     # Target (excluding the first 3 rows)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'min_samples_split': [1, 2, 3],
    'min_samples_leaf': [10, 11, 12],
    'max_features': [2, 3, 4]
}

# Instantiate the model
model = RandomForestRegressor(random_state=0)

# Instantiate Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit Grid Search to training data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model obtained from Grid Search
best_model = grid_search.best_estimator_

# Predictions
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

# Model evaluation
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

print('Train Score:', train_score)
print('Test Score:', test_score)

# Prediction for the last 3 rows
new_data = merged_data[['bitcoin_smoothed', 'ethereum_smoothed', 'monero_smoothed']].tail(3)
predictions = best_model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(merged_data['bitcoin_smoothed'].tail(3))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(merged_data.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(merged_data.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('Bitcoin Searches Forecast (Smoothed)' if apply_smoothing_flag else 'Bitcoin Searches Forecast (Original)')
plt.xlabel('Week')
plt.ylabel('Bitcoin Searches (Smoothed)' if apply_smoothing_flag else 'Bitcoin Searches (Original)')
plt.legend()
plt.show()
