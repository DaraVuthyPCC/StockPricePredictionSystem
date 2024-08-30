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
                          test_size, price_value, random_state, data_file, feature_columns):
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

    result["df"].to_csv(data_file)

    return result


# candlestick chart function that take in dataframe and number of trading days as parameters
def candlestick_chart(df, n):
    # if the number of trading days is more than one, we change the features column into first, max, min, last
    # else just keep it the same
    if n > 1:
        df_resampled = df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
    else:
        df_resampled = df.copy()

    fig = go.Figure(data=[go.Candlestick(
        x=df_resampled.index,
        open=df_resampled['Open'],
        high=df_resampled['High'],
        low=df_resampled['Low'],
        close=df_resampled['Close'],
        increasing_line_color='green',  # Color for increasing candles
        decreasing_line_color='red',    # Color for decreasing candles
        name='Candlestick'
    )])

    fig.add_trace(go.Scatter(
        x=[df_resampled.index[-1], next_day],
        y=[df_resampled['Close'].iloc[-1], prediction[0][0]],
        mode='lines+markers',
        line=dict(color='orange', width=2),
        marker=dict(size=8),
        name='Next Day Prediction'
    ))

    fig.update_layout(
        title=f'Stock Price Candlestick Chart ({n} Trading Day(s) per Candle)',
        xaxis_title='Date',
        yaxis_title='Price',
        # xaxis_rangeslider_visible=False,
        template='plotly_dark',
        width=1200,
        height=600,
    )

    fig.show()

# graphing boxplot chart
def boxplot_chart(df, n, step=5):
    if n == 1:
        rolling_windows = [df['Close'].values]
    else:
        rolling_windows = [df['Close'][i:i + n].values for i in range(0, len(df) - n + 1)]

    window_indices = range(len(rolling_windows))

    plt.figure(figsize=(12, 6))
    plt.boxplot(rolling_windows, showfliers=False)

    plt.xticks(ticks=np.arange(1, len(window_indices) + 1, step=step),
               labels=np.arange(1, len(window_indices) + 1, step=step))

    plt.title(f'Boxplot of {COMPANY} Stock Prices Over a Moving Window of {n} Trading Day(s)')
    plt.xlabel('Rolling Window Index')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

# creating a model
def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, 
                dropout=0.3, loss="mean_absolute_error", optimizer="adam"):
    """
        sequence_length: the number of steps that the model take in each input
        n_features: the number of features
        units: the number of units in each LSTM cell which determine the dimensionality of output space of each layer
        cell: the type of networks to use
        n_layers: the number of stacked layer
        dropout: the dropout rate after each layer to prevent overfitting
        loss: the loss function that is used to measure error
        optimizer: the optimizer that is used to minimize the loss function
    """
    # a sequential model which has many layer
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(cell(units, return_sequences=True, input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            model.add(cell(units, return_sequences=False))
        else:
            model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    return model

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
                             SPLIT_BY_DATE, TEST_SIZE, PRICE_VALUE, RANDOM_STATE, data_file, FEATURE_COLUMNS)

N_STEPS = 50
UNITS = 256
CELL = LSTM
N_LAYERS = 4
DROPOUT = 0.3
LOSS = "mean_absolute_error"
OPTIMIZER = "adam"
EPOCHS = 50
BATCH = 64

# Building model
model_dir = 'model'
# Model name is gonna be saved based on the input we get from all the variable that we have set
model_file = f'{model_dir}/{COMPANY}-{N_STEPS}-{UNITS}-{CELL.__name__}-{N_LAYERS}-{DROPOUT}-{LOSS}-{OPTIMIZER}-{EPOCHS}-{BATCH}_model.keras'

# check if a model folder already exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# check if a model already exists
if os.path.isfile(model_file):
    model = load_model(model_file)
else:
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), UNITS, CELL, N_LAYERS,
                         DROPOUT, LOSS, OPTIMIZER)
    model.fit(data["X_train"], data["y_train"], epochs=EPOCHS, batch_size=BATCH)
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

n = 3 # number of trading days

fig = candlestick_chart(df, n)
fig1 = boxplot_chart(df, n)
