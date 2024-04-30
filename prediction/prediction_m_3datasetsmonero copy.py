import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import wiener


def apply_smoothing(data):
    return wiener(data)


df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
df = df.dropna()

df['Week'] = pd.to_datetime(df['Week'])

df_ethereum = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)
df_monero = pd.read_csv('Google Trends Data Challenge Datasets/trends/monero.csv', skiprows=1)

df_ethereum = df_ethereum.dropna()
df_monero = df_monero.dropna()

df_ethereum['Week'] = pd.to_datetime(df_ethereum['Week'])
df_monero['Week'] = pd.to_datetime(df_monero['Week'])

apply_smoothing_flag = True  
if apply_smoothing_flag:
    df['bitcoin_smoothed'] = apply_smoothing(df['bitcoin: (Worldwide)'])
    df_ethereum['ethereum_smoothed'] = apply_smoothing(df_ethereum['ethereum: (Worldwide)'])
    df_monero['monero_smoothed'] = apply_smoothing(df_monero['monero: (Worldwide)'])
else:
    df['bitcoin_smoothed'] = df['bitcoin: (Worldwide)']
    df_ethereum['ethereum_smoothed'] = df_ethereum['ethereum: (Worldwide)']
    df_monero['monero_smoothed'] = df_monero['monero: (Worldwide)']

df.set_index('Week', inplace=True)
df_monthly = df.resample('3W').mean()
df_ethereum.set_index('Week', inplace=True)
df_ethereum_monthly = df_ethereum.resample('3W').mean()
df_monero.set_index('Week', inplace=True)
df_monero_monthly = df_monero.resample('3W').mean()

df_monthly = df_monthly.merge(df_ethereum_monthly[['ethereum_smoothed']], on='Week', how='left')
df_monthly = df_monthly.merge(df_monero_monthly[['monero_smoothed']], on='Week', how='left')

df_monthly['ethereum_smoothed'].fillna(method='ffill', inplace=True)
df_monthly['monero_smoothed'].fillna(method='ffill', inplace=True)

X = df_monthly[['bitcoin_smoothed', 'ethereum_smoothed', 'monero_smoothed']][:-3] 
y = df_monthly['bitcoin_smoothed'][3:] 

print(df_monthly)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [125],
    'max_depth': [None],
    'min_samples_split': [3],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

print('Train Score:', train_score)
print('Test Score:', test_score)

new_data = df_monthly[['bitcoin_smoothed', 'ethereum_smoothed', 'monero_smoothed']].tail(3)
predictions = best_model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(df_monthly['bitcoin_smoothed'].tail(3))

plt.figure(figsize=(10, 6))
plt.plot(df_monthly.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(df_monthly.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(df_monthly.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('bitcoin Searches Forecast (Smoothed)' if apply_smoothing_flag else 'bitcoin Searches Forecast (Original)')
plt.xlabel('Month')
plt.ylabel('bitcoin Searches (Smoothed)' if apply_smoothing_flag else 'bitcoin Searches (Original)')
plt.legend()
plt.show()
