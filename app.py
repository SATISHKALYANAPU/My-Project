import pandas as pd
import numpy as np
import requests
# Step1:  Fetch Real-Time Stock Data
# Replace with your Alpha Vantage API key
API_KEY = 'U92BV1235KTDJBV9'

def fetch_stock_data(symbol, interval='daily', output_size='compact'):
    """
    Fetch stock data from Alpha Vantage API.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{interval.upper()}&symbol={symbol}&apikey={API_KEY}&outputsize={output_size}'
    response = requests.get(url)
    data = response.json()

    # Extract time series data
    time_series = data[f'Time Series ({interval.capitalize()})']
    df = pd.DataFrame(time_series).T
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    return df

# Fetch data for a stock (e.g., Apple)
df = fetch_stock_data('AAPL')
print(df.head())


# Step:2 Step 3: Clean and Preprocess Data
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Calculate moving averages
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()

# Display cleaned data
print(df.tail())



#Step 4: Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA_50'], label='50-Day MA')
plt.plot(df['MA_200'], label='200-Day MA')
plt.title('Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot candlestick chart
import mplfinance as mpf

mpf.plot(df.tail(30), type='candle', style='charles', volume=True, title='Candlestick Chart (Last 30 Days)')


# Step 5: Predictive Modeling
# ARIMA Model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Split data into train and test sets
train_data = df['Close'][:-30]
test_data = df['Close'][-30:]

# Fit ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# Make predictions
predictions = model_fit.forecast(steps=30)

# Evaluate model
mse = mean_squared_error(test_data, predictions)
print(f'Mean Squared Error: {mse}')

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.title('ARIMA Model Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Create training dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Make predictions
test_data = scaled_data[-time_step:]
X_test = test_data.reshape(1, time_step, 1)
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
print(f'Predicted Price: {predicted_price[0][0]}')

# Step 6: Interactive Dashboard with Streamlit
import streamlit as st
import yfinance as yf

# Streamlit app
st.title('Stock Market Analysis Dashboard')

# Sidebar for user input
symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL')

# Fetch data
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')

# Check if the DataFrame has a MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    # Simplify the DataFrame for a single ticker
    data = data.droplevel(level=1, axis=1)

# Calculate moving averages
if len(data) >= 50:
    data['MA_50'] = data['Close'].rolling(window=50).mean()
if len(data) >= 200:
    data['MA_200'] = data['Close'].rolling(window=200).mean()

# Display data
st.subheader('Stock Data')
st.write(data)

# Plot closing price
st.subheader('Closing Price')
st.line_chart(data['Close'])

# Plot moving averages (if available)
if 'MA_50' in data.columns and 'MA_200' in data.columns:
    st.subheader('Moving Averages')
    st.line_chart(data[['Close', 'MA_50', 'MA_200']])
else:
    st.warning("Insufficient data to calculate moving averages.")