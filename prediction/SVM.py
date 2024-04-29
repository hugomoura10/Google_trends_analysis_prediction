import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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

 
apply_smoothing_flag = True   
if apply_smoothing_flag:
     
    merged_data['bitcoin_smoothed'] = apply_smoothing(merged_data['bitcoin: (Worldwide)'])
else:
    merged_data['bitcoin_smoothed'] = merged_data['bitcoin: (Worldwide)']

 
X = merged_data[['bitcoin_smoothed', 'ethereum: (Worldwide)']][:-3]   
y = merged_data['bitcoin_smoothed'][3:]      

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

 
model = SVR()

 
model.fit(X_train, y_train)

 
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

 
 

 
new_data = merged_data[['bitcoin_smoothed', 'ethereum: (Worldwide)']].tail(3)
predictions = model.predict(new_data)
print('The model predicts the last 3 rows:', predictions)
print('Actual values for the last 3 rows:')
print(merged_data['bitcoin_smoothed'].tail(3))

 
plt.figure(figsize=(10, 6))
plt.plot(merged_data.index[3:], y, label='Actual bitcoin Searches (Smoothed)')
plt.plot(merged_data.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(merged_data.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('Bitcoin Searches Forecast (Smoothed)' if apply_smoothing_flag else 'Bitcoin Searches Forecast (Original)')
plt.xlabel('Week')
plt.ylabel('Bitcoin Searches (Smoothed)' if apply_smoothing_flag else 'Bitcoin Searches (Original)')
plt.legend()
plt.show()
