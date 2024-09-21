# import necessary package
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
import time

from datetime import timedelta, datetime as dt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import huber
from tensorflow.keras.regularizers import l2
from collections import deque

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

def load_and_process_data(company, n_steps=50, scale=True, lookup_step=1, test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low']):
    """
    Params:
        
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

    df = yf.download(company, TRAIN_START, TRAIN_END)

    df = df.interpolate().dropna()
    # all the value we return from this function
    result = {}
    # make a copy of the original data
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    train_samples = int((1 - test_size) * len(X))
    result["X_train"] = X[:train_samples]
    result["y_train"] = y[:train_samples]
    result["X_test"]  = X[train_samples:]
    result["y_test"]  = y[train_samples:]

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

def create_model(prediction_days, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(cell(units, return_sequences=True, input_shape=(prediction_days, n_features)))
        elif i == n_layers - 1:
            model.add(cell(units, return_sequences=False))
        else:
            model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    return model

COMPANY = 'AAPL'
PREDICTION_DAYS = 50
SCALE = True
FUTURE = 15
TEST_SIZE = 0.2
PRICE_VALUE = "Close"
FEATURE_COLUMNS2 = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "MA50", "MA200", "RSI", "Momentum"]
FEATURE_COLUMNS = ["Close", "Volume", "Open", "High", "Low"]
FEATURE_COLUMNS1 = ["Close"]
date_now = time.strftime("%Y-%m-%d")

data_file = f"{data_dir}/{COMPANY}-{date_now}-{PREDICTION_DAYS}-{SCALE}-{PRICE_VALUE}-{len(FEATURE_COLUMNS)}-{FUTURE}.csv"

data = load_and_process_data(COMPANY, PREDICTION_DAYS, SCALE, FUTURE, TEST_SIZE, FEATURE_COLUMNS)

N_STEPS = 100
UNITS = 1024
CELL = LSTM
N_LAYERS = 4
DROPOUT = 0.5
LOSS = "mean_squared_error"
EPOCHS = 100
BATCH = 128
ACTIVATION = "linear"
OPTIMIZER = "adam"

# Building model
model_dir = 'model'
# Model name is gonna be saved based on the input we get from all the variable that we have set
model_file = f'{model_dir}/{COMPANY}-{N_STEPS}-{UNITS}-{CELL.__name__}-{N_LAYERS}-{DROPOUT}-{LOSS}-{OPTIMIZER}-{EPOCHS}-{BATCH}-{ACTIVATION}-{len(FEATURE_COLUMNS)}_model.keras'

# check if a model folder already exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# check if a model already exists
if os.path.isfile(model_file):
    model = load_model(model_file)
else:
    model = create_model(PREDICTION_DAYS, len(FEATURE_COLUMNS), UNITS, CELL, N_LAYERS, DROPOUT, LOSS, OPTIMIZER)
    model.fit(data["X_train"], data["y_train"], epochs=EPOCHS, batch_size=BATCH,
            validation_data=(data["X_test"], data["y_test"]))
    model.save(model_file)

X_test = data["X_test"]
y_test = data["y_test"]

y_pred = model.predict(X_test)
if SCALE:
    y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))

test_df = data["test_df"]
# add predicted future prices to the dataframe
test_df[f"close_{FUTURE}"] = y_pred
# add true future prices to the dataframe
test_df[f"true_close_{FUTURE}"] = y_test
# sort the dataframe by date
test_df.sort_index(inplace=True)
final_df = test_df

last_sequence = data["last_sequence"][-N_STEPS:]
# expand dimension
last_sequence = np.expand_dims(last_sequence, axis=0)
# get the prediction (scaled from 0 to 1)
prediction = model.predict(last_sequence)
# get the price (by inverting the scaling)
if SCALE:
    predicted_price = data["column_scaler"]["Close"].inverse_transform(prediction)[0][0]
else:
    predicted_price = prediction[0][0]

plt.figure(figsize=(30,10))
plt.plot(final_df[f'true_close_{FUTURE}'], c='b')
plt.plot(final_df[f'close_{FUTURE}'], c='r')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend(["Actual Price", "Predicted Price"])
plt.show()

# fig = candlestick_chart(df, n, predicted_values, FUTURE)
# fig1 = candlestick_chart()
# fig1 = boxplot_chart(df, n)
