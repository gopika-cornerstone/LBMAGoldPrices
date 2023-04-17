import pandas as pd
import tensorflow as tf
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

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

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
