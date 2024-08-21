# import necessary package
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
import time

from datetime import timedelta, datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Create data folder if it does not exist
data_dir = 'data'

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# check if date is in the right format
def check_date(prompt):
    while True:
        date_input = input(prompt)
        try:
            date_obj = dt.strptime(date_input, "%Y-%m-%d")
            return date_obj
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

def load_and_process_data(company, predict_day, scale, shuffle, future, split_by_date,
                          test_size, price_value, random_state, feature_columns):
    """
        company: we are going to use apple stock price for this assignment which is 'AAPL'
        predict_days: number of days to look back to base the prediction which i choose 60
        scale: to scale the data between 0 and 1 which is set to true
        shuffle: set to false as we want the data to be in order to get a more accurate prediction
        future: the future to be predicted which is 1
        test_size: the size of the data that is use to be tested which is 20%
        feature_columns: list of features to use to feed into model
    """
    # asking for date input from user
    while True:
        TRAIN_START = check_date("Please enter a start date for reading(YYYY-MM-DD): ")
        TRAIN_END = check_date("Please enter an end date for reading(YYYY-MM-DD): ")
        # check if end date is later than start date
        if TRAIN_END > TRAIN_START:
            break
        else:
            print("Error: End Date must be later than Start Date. Please try again.")

    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(company, str):
        # load it from yfinance library
        df = yf.download(company, TRAIN_START, TRAIN_END)
    elif isinstance(company, pd.DataFrame):
        # already loaded, use it directly
        df = company
    else:
        raise TypeError("ticker can be either a str or a pd.DataFrame instances")

    df = df.interpolate().dropna()
    # all the value we return from this function
    result = {}
    # make a copy of the original data
    result['df'] = df.copy()

    # make sure every columns from feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a columm
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data from -1 to 1
        for column in feature_columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by future
    df['future'] = df[price_value].shift(-future)

    # dropping NaNs
    df.dropna(inplace=True)
    # filling NaNs with interpolation
    df.interpolate()

    # To store the training data
    x_train = []
    y_train = []

    # Prepare the data
    for x in range(predict_day, len(df)):
        x_train.append(df[feature_columns].values[x-predict_day:x])
        y_train.append(df['future'].iloc[x])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], predict_day, len(feature_columns)))

    if split_by_date:
        train_samples = int((1 - test_size) * len(x_train))
        result.update({"X_train": x_train[:train_samples], "y_train": y_train[:train_samples],
                       "X_test": x_train[train_samples:], "y_test": y_train[train_samples:]})
    else:
        X_train, X_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=test_size, shuffle=shuffle,
                                                            random_state=random_state)
        result.update({"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test})

    return result

COMPANY = 'AAPL'
PREDICTION_DAYS = 60 
SCALE = True
SHUFFLE = False
FUTURE = 1
SPLIT_BY_DATE = True
TEST_SIZE = 0.2
PRICE_VALUE = "Close"
RANDOM_STATE = 344
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
date_now = time.strftime("%Y-%m-%d")

data_file = f"{data_dir}/{COMPANY}-{date_now}-{SCALE}-{SPLIT_BY_DATE}-{PRICE_VALUE}.csv"

data = load_and_process_data(COMPANY, PREDICTION_DAYS, SCALE, SHUFFLE, FUTURE, 
                             SPLIT_BY_DATE, TEST_SIZE, PRICE_VALUE, RANDOM_STATE, FEATURE_COLUMNS)

data["df"].to_csv(data_file)

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
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(PREDICTION_DAYS, len(FEATURE_COLUMNS))),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    model.fit(data["X_train"], data["y_train"], epochs=50, batch_size=64)
            #   validation_split=0.2, callbacks=[early_stopping, reduce_lr])
    model.save(model_file)

# Load the test data
TEST_START = '2024-01-01'
TEST_END = '2024-08-15'

test_data_file = f"{data_dir}/{COMPANY}_test.csv"

test_data = yf.download(COMPANY, TEST_START, TEST_END)

test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values
total_dataset = pd.concat((data["df"][PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = model_inputs.reshape(-1, 1)
scaler = data["column_scaler"][PRICE_VALUE]
model_inputs = scaler.transform(model_inputs)
test_data.to_csv(test_data_file)

x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)

# Total elements in x_test
total_elements = x_test.shape[0]  # Total rows in x_test
num_samples = total_elements // PREDICTION_DAYS  # Correctly calculate number of samples

# Trim x_test to make it divisible by PREDICTION_DAYS
if total_elements % PREDICTION_DAYS != 0:
    x_test = x_test[:num_samples * PREDICTION_DAYS]

# Ensure num_features is the number of features used in your model
num_features = len(FEATURE_COLUMNS)  # Assuming you want to use all feature columns

# Reshape x_test to (num_samples, PREDICTION_DAYS, num_features)
# Ensure that the resulting shape matches the number of features you have
x_test = np.reshape(x_test, (-1, PREDICTION_DAYS, num_features))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data).reshape(-1, PREDICTION_DAYS)

# If you need to match the shape (1, 60, 6), you can tile the data to create 6 identical features
if real_data.shape[1] == PREDICTION_DAYS:
    real_data = np.tile(real_data, (1, 6)).reshape(1, PREDICTION_DAYS, len(FEATURE_COLUMNS))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

last_date = test_data.index[-1]
next_day = last_date + timedelta(days=1)

df = pd.read_csv(test_data_file, index_col='Date', parse_dates=True)
df_future = pd.DataFrame([prediction[0][0]], index=[next_day], columns=[PRICE_VALUE])

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
