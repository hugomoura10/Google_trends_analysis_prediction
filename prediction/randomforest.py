import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('Google Trends Data Challenge Datasets/trends/bitcoin.csv', skiprows=1)
df = df.dropna()

df['Week'] = pd.to_datetime(df['Week'])

df.plot(x="Week", y="bitcoin: (Worldwide)")
plt.xticks(rotation=45)
plt.show()

X = df[['bitcoin: (Worldwide)']][:-1]  # Features (excluding the last row)
y = df['bitcoin: (Worldwide)'][1:]    # Target (excluding the first row)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print('Train Score:', train_score)
print('Test Score:', test_score)

# Step 7: Prediction
new_data = df[['bitcoin: (Worldwide)']].tail(1)
prediction = model.predict(new_data)
print('The model predicts the last row:', prediction)
print('Actual value:', df['bitcoin: (Worldwide)'].iloc[-1])

plt.figure(figsize=(10, 6))
plt.plot(df.index[1:], y, label='Actual Bitcoin Searches')
plt.plot(df.index[1:len(train_pred) + 1], train_pred, label='Train Predictions')
plt.plot(df.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('Bitcoin Searches Forecast')
plt.xlabel('Week')
plt.ylabel('Bitcoin Searches')
plt.legend()
plt.show()