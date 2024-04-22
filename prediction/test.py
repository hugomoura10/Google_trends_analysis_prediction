import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load merged dataset with both search trend and price data
df = pd.read_csv('bitcoin_merged_data.csv')
df = df.dropna()
df['Week'] = pd.to_datetime(df['Week'])

df.plot(x="Week", y="bitcoin: (Worldwide)")
plt.xticks(rotation=45)
plt.show()

# Features and target variable
X = df[['bitcoin: (Worldwide)', 'Close']][:-3]  # Features (excluding the last three rows)
y = df['bitcoin: (Worldwide)'][3:]    # Target (excluding the first three rows)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
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

# Prediction for the last three rows
new_data = df[['bitcoin: (Worldwide)', 'Close']].tail(3)
predictions = model.predict(new_data)
print('The model predicts the last three rows:', predictions)
print('Actual values:', df['bitcoin: (Worldwide)'].iloc[-3:])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df.index[3:], y, label='Actual Bitcoin Searches')
plt.plot(df.index[3:len(train_pred) + 3], train_pred, label='Train Predictions')
plt.plot(df.index[-len(test_pred):], test_pred, label='Test Predictions')
plt.title('Bitcoin Searches Forecast')
plt.xlabel('Week')
plt.ylabel('Bitcoin Searches')
plt.legend()
plt.show()
