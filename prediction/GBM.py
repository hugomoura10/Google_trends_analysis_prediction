import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from scipy.signal import wiener

def apply_smoothing(data):
    return wiener(data)

bitcoin_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
bitcoin_trend_df = bitcoin_trend_df.dropna()
bitcoin_trend_df['Week'] = pd.to_datetime(bitcoin_trend_df['Week'])

ethereum_trend_df = pd.read_csv('Google Trends Data Challenge Datasets/trends/ethereum.csv', skiprows=1)
ethereum_trend_df['Week'] = pd.to_datetime(ethereum_trend_df['Week'])

merged_data = pd.merge(bitcoin_trend_df, ethereum_trend_df, on='Week', how='left')

apply_smoothing_flag = True  # Set this to False if you don't want to apply smoothing
if apply_smoothing_flag:
    merged_data['bitcoin_smoothed'] = apply_smoothing(merged_data['bitcoin: (Worldwide)'])
else:
    merged_data['bitcoin_smoothed'] = merged_data['bitcoin: (Worldwide)']

X = merged_data[['bitcoin_smoothed', 'ethereum: (Worldwide)']][:-3] 
y = merged_data['bitcoin_smoothed'][3:]    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

model = XGBRegressor()

param_grid = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.1],
    'min_child_weight': [1],
    'subsample': [1],
    'colsample_bytree': [1],
    'gamma': [0]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

best_model_train_pred = best_model.predict(X_train)
best_model_test_pred = best_model.predict(X_test)

best_model_train_score = best_model.score(X_train, y_train)
best_model_test_score = best_model.score(X_test, y_test)

print('Best Model Train Score:', best_model_train_score)
print('Best Model Test Score:', best_model_test_score)

# Prediction for the last 3 rows
new_data = merged_data[['bitcoin_smoothed', 'ethereum: (Worldwide)']].tail(3)
predictions = best_model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(merged_data['bitcoin_smoothed'].tail(3))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(merged_data.index[3:len(best_model_train_pred) + 3], best_model_train_pred, label='Train Predictions')
plt.plot(merged_data.index[-len(best_model_test_pred):], best_model_test_pred, label='Test Predictions')
plt.title('Bitcoin Searches Forecast (Smoothed)' if apply_smoothing_flag else 'Bitcoin Searches Forecast (Original)')
plt.xlabel('Week')
plt.ylabel('Bitcoin Searches (Smoothed)' if apply_smoothing_flag else 'Bitcoin Searches (Original)')
plt.legend()
plt.show()
