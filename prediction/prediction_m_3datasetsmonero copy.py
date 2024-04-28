import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Function to apply Wiener smoothing
def apply_smoothing(data):
    return wiener(data)

# Load data
df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
df = df.dropna()

# Convert 'Week' column to datetime
df['Week'] = pd.to_datetime(df['Week'])

# Load additional datasets
df_ethereum = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)
df_monero = pd.read_csv('Google Trends Data Challenge Datasets/trends/monero.csv', skiprows=1)

# Drop NaN values
df_ethereum = df_ethereum.dropna()
df_monero = df_monero.dropna()

# Convert 'Week' column to datetime
df_ethereum['Week'] = pd.to_datetime(df_ethereum['Week'])
df_monero['Week'] = pd.to_datetime(df_monero['Week'])

# Choose whether to apply smoothing or not
apply_smoothing_flag = True  # Set this to False if you don't want to apply smoothing
if apply_smoothing_flag:
    # Apply Wiener smoothing to the data
    df['bitcoin_smoothed'] = apply_smoothing(df['bitcoin: (Worldwide)'])
    df_ethereum['ethereum_smoothed'] = apply_smoothing(df_ethereum['ethereum: (Worldwide)'])
    df_monero['monero_smoothed'] = apply_smoothing(df_monero['monero: (Worldwide)'])
else:
    df['bitcoin_smoothed'] = df['bitcoin: (Worldwide)']
    df_ethereum['ethereum_smoothed'] = df_ethereum['ethereum: (Worldwide)']
    df_monero['monero_smoothed'] = df_monero['monero: (Worldwide)']

# Resample data to monthly frequency
df.set_index('Week', inplace=True)
df_monthly = df.resample('3W').mean()
df_ethereum.set_index('Week', inplace=True)
df_ethereum_monthly = df_ethereum.resample('3W').mean()
df_monero.set_index('Week', inplace=True)
df_monero_monthly = df_monero.resample('3W').mean()

# Merge datasets based on 'Week' column
df_monthly = df_monthly.merge(df_ethereum_monthly[['ethereum_smoothed']], on='Week', how='left')
df_monthly = df_monthly.merge(df_monero_monthly[['monero_smoothed']], on='Week', how='left')

# Fill NaN values with appropriate methods
df_monthly['ethereum_smoothed'].fillna(method='ffill', inplace=True)
df_monthly['monero_smoothed'].fillna(method='ffill', inplace=True)

# Prepare data for modeling
X = df_monthly[['bitcoin_smoothed', 'ethereum_smoothed', 'monero_smoothed']][:-3]  # Features (excluding the last 3 rows)
y = df_monthly['bitcoin_smoothed'][3:]  # Target (excluding the first 3 rows)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Instantiate the model
model = RandomForestRegressor(random_state=42)

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
new_data = df_monthly[['bitcoin_smoothed', 'ethereum_smoothed', 'monero_smoothed']].tail(3)
predictions = best_model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(df_monthly['bitcoin_smoothed'].tail(3))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df_monthly.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(df_monthly.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(df_monthly.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('bitcoin Searches Forecast (Smoothed)' if apply_smoothing_flag else 'bitcoin Searches Forecast (Original)')
plt.xlabel('Month')
plt.ylabel('bitcoin Searches (Smoothed)' if apply_smoothing_flag else 'bitcoin Searches (Original)')
plt.legend()
plt.show()
