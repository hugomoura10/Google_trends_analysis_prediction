import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import wiener
from datetime import timedelta

def apply_smoothing(data):
    return wiener(data)

df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
df = df.dropna()

df['Week'] = pd.to_datetime(df['Week'])

apply_smoothing_flag = True  
if apply_smoothing_flag:
    df['bitcoin_smoothed'] = apply_smoothing(df['bitcoin: (Worldwide)'])
else:
    df['bitcoin_smoothed'] = df['bitcoin: (Worldwide)']

X = df[['bitcoin_smoothed']][:-3]  
y = df['bitcoin_smoothed'][3:]    

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
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

new_data = df[['bitcoin_smoothed']].tail(3)
predictions = best_model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(df['bitcoin_smoothed'].tail(3))

last_date = df.index[-1] 
future_dates = pd.date_range(start=last_date + timedelta(days=7), end='2024-12-31', freq='W')
future_dates = pd.date_range(start=last_date + pd.DateOffset(7), end='2024-12-31', freq='W')

future_dates_2d = future_dates.values.reshape(-1, 1)

future_predictions = best_model.predict(future_dates_2d)

print('Predictions for the future (until the end of 2024):')
print(future_predictions)

plt.figure(figsize=(10, 6))
plt.plot(df.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(df.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(df.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
plt.title('bitcoin Searches Forecast (Smoothed)' if apply_smoothing_flag else 'bitcoin Searches Forecast (Original)')
plt.xlabel('Week')
plt.ylabel('bitcoin Searches (Smoothed)' if apply_smoothing_flag else 'bitcoin Searches (Original)')
plt.legend()
plt.show()
