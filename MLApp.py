import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the cleaned dataset into a DataFrame
df = pd.read_csv('/Users/gopikasriram/Desktop/LBMAGold/clean.csv', index_col='Date')

# Convert the index to a pandas datetime object
df.index = pd.to_datetime(df.index)

# Create a Streamlit app
st.title('Gold Price Prediction')
st.write('Choose a model and a time period range to view predicted vs actual gold prices in 3 different currencies.')

# Allow the user to choose a model
model = st.sidebar.selectbox(
    'Choose a model:',
    ('Random Forest', 'Linear Regression', 'Neural Network')
)



# Allow the user to choose a time period range
start_year = st.sidebar.slider('Start Year', 1999, 2022,2015)
end_year = st.sidebar.slider('End Year', start_year, 2023, 2022)

# Filter the data based on the user's time period range
start_date = pd.to_datetime(str(start_year))
end_date = pd.to_datetime(str(end_year))
mask = (df.index >= start_date) & (df.index <= end_date)
df_filtered = df.loc[mask]

currency_options = ['USD (Average)', 'EURO (Average)', 'GBP (Average)']
currency = st.sidebar.selectbox(
    'Choose a currency:',
    currency_options
)

# Split the data into training and testing sets
X = df_filtered.drop(columns=[currency]) # Use all columns except 'USD (Average)' as features
y = df_filtered[currency] # Use 'USD (Average)' as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the selected model
if model == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model == 'Linear Regression':
    model = LinearRegression()
elif model == 'Neural Network':
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = r2_score(y_test, y_pred)

# Plot predicted prices vs actual prices
predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
y_test = y_test.sort_index() 
predicted_price.plot(figsize=(20, 7))
y_test.plot()
plt.legend(['predicted price', 'actual_price'])
plt.ylabel("LBMA Gold Prices")
plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
# Display the accuracy score and plot
st.write('Accuracy:', accuracy)
st.write('Currency:', currency[0:4])
st.pyplot()

# Convert the index to a pandas datetime object
f = pd.DataFrame()
f.index = pd.to_datetime(f.index)

f['ds'] = df.index.get_level_values('Date')
f['y'] = df['USD (Average)'].values

m = Prophet()

# Set the uncertainty interval to 95% (the Prophet default is 80%)
m.interval_width = 0.95

# Fit the model to the historical data
model = m.fit(f)

# Set up a Streamlit slider for choosing the prediction time
prediction_time = st.slider("Select the number of years for prediction of Gold Prices(USD):", min_value=1, max_value=5, value=3, step=1)

# Make a future dataframe for the desired prediction time
future = m.make_future_dataframe(periods=prediction_time, freq='Y')

# Make predictions for the future data
forecast = m.predict(future)

# Create a plot of the forecast
f_forecast = forecast.copy()
f0 = df.copy()

trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(f_forecast['ds']),
    y = list(f_forecast['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
)
upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(f_forecast['ds']),
    y = list(f_forecast['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty'
)
lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(f_forecast['ds']),
    y = list(f_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)
tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(f['ds']),
   y = list(f['y']),
   marker=dict(
      color='white',
      line=dict(width=2)
   )
)
data = [trace1, lower_band, upper_band,tracex]

layout = dict(title='GOLD PRICE PREDICTION',
             xaxis=dict(title = 'Year', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)

# Display the plot using Streamlit
st.plotly_chart(figure)

# Plot the forecast using the Prophet plot method
plot1 = m.plot(forecast)
plt.show()