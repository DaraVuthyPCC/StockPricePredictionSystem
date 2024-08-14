# import necessary package
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, Input

# Create data folder if it does not exist
data_dir = 'data'

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# DATA_SOURCE = "yahoo"
COMPANY = 'AAPL'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo

# Check if data exists
data_file = f"{data_dir}/{COMPANY}.csv"

# Getting data
if os.path.isfile(data_file):
    data = pd.read_csv(data_file, index_col='Date', parse_dates=True)  # reading an already existing data
else:
    data = yf.download(COMPANY, TRAIN_START, TRAIN_END)  # download data from yahoo finance
    data.to_csv(data_file)  # saving the data to into a file

# Check if a prepared data file exists
prepared_data_file = f"{data_dir}/{COMPANY}_prepared.csv"
PRICE_VALUE = "Close"

# Preparing data
scaler = MinMaxScaler(feature_range=(-1, 1))

if os.path.isfile(prepared_data_file):
    data = pd.read_csv(prepared_data_file, index_col='Date', parse_dates=True)
    scaled_data = data[PRICE_VALUE].values
else:
    scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
    data[PRICE_VALUE] = scaled_data
    scaled_data = scaled_data[:, 0]
    data.to_csv(prepared_data_file)

# Number of days to look back to base the prediction
PREDICTION_DAYS = 120  # Original

# To store the training data
x_train = []
y_train = []

# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Check for x and y array
x_train_file = os.path.join(f"{data_dir}/x_train.npy")
y_train_file = os.path.join(f"{data_dir}/y_train.npy")

if not os.path.exists(x_train_file) and os.path.exists(y_train_file):
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)
else:
    x_train, y_train = np.array(x_train), np.array(y_train)
    np.save(x_train_file, x_train)
    np.save(y_train_file, y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building model
model_dir = 'model'
model_file = f'{model_dir}/{COMPANY}_model.keras'

# check if a model folder already exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# check if a model already exists
if os.path.isfile(model_file):
    model = load_model(model_file)
else:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=64)
    model.save(model_file)

# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-08-09'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

test_data_file = f"{data_dir}/{COMPANY}_test.csv"
prepared_test_data_file = f"{data_dir}/{COMPANY}_prepared_test.csv"

test_data = yf.download(COMPANY, TEST_START, TEST_END)
test_data.to_csv(test_data_file)

test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values
total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)
test_data.to_csv(prepared_test_data_file)

x_test, y_test = [], actual_prices
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

last_date = test_data.index[-1]
next_day = last_date + dt.timedelta(days=1)

df = pd.read_csv(prepared_test_data_file, index_col='Date', parse_dates=True)
df_future = pd.DataFrame([prediction[0][0]], index=[next_day], columns=['Close'])

# Candlesticks chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',  # Color for increasing candles
    decreasing_line_color='red',    # Color for decreasing candles
    name='Candlestick'
)])

fig.add_trace(go.Scatter(
    x=[df.index[-1], next_day],
    y=[df['Close'].iloc[-1], prediction[0][0]],
    mode='lines+markers',
    line=dict(color='orange', width=2),
    marker=dict(size=8),
    name='Next Day Prediction'
))

fig.update_layout(
    title='Stock Price Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    width=1200,
    height=600,
)

fig.show()

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()
