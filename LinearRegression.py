import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the cleaned dataset into a DataFrame
df = pd.read_csv('/Users/gopikasriram/Desktop/LBMAGold/clean.csv', index_col='Date')

# Sort the DataFrame by the index to ensure the data is in chronological order
df = df.sort_index()

# Split the data into training and testing sets
X = df.drop(columns=['USD (Average)']) # Use all columns except 'USD (Average)' as features
y = df['USD (Average)'] # Use 'USD (Average)' as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model performance
accuracy = r2_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Plot predicted prices vs actual prices
predicted_price = pd.DataFrame(
    y_pred, index=y_test.index, columns=['price'])
predicted_price = predicted_price.sort_index() # Sort by index
y_test = y_test.sort_index() # Sort by index
plt.plot(y_test.index, y_test, label='Actual Prices')
plt.plot(predicted_price.index, predicted_price, label='Predicted Prices')
plt.xlabel('Year')
plt.ylabel('USD (Average)')
plt.legend()
plt.show()
