# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
import time
from datetime import timedelta, datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from collections import deque
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import itertools  # Add this for the combinations of batch size, epochs, and length

# Create data directory if it doesn't exist
data_dir = 'data'
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# Function to validate date input
def check_date(prompt):
    while True:
        date_input = input(prompt)
        try:
            date_obj = dt.strptime(date_input, "%Y-%m-%d")
            return date_obj
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

# Function to load and process data
def load_and_process_data(company, n_steps, scale, lookup_step, test_size, feature_columns):
    # Request start and end date inputs from user
    while True:
        train_start = check_date("Please enter a start date for reading (YYYY-MM-DD): ")
        train_end = check_date("Please enter an end date for reading (YYYY-MM-DD): ")
        if train_end > train_start:
            break
        else:
            print("Error: End Date must be later than Start Date. Please try again.")

    # Load data using yfinance
    df = yf.download(company, train_start, train_end)
    df = df.interpolate().dropna()  # Interpolate missing values and drop remaining NaNs

    # Validate feature columns
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # Add date column if not present
    if "Date" not in df.columns:
        df["Date"] = df.index

    # Copy the original data for safe keeping
    result = {'df': df.copy()}

    # Scale the data if required
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # Add future column for prediction by shifting the 'close' value up by the amount of lookup_step
    df['future'] = df['Close'].shift(-lookup_step)
    
    # Capture the last lookup_step before dropping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)  # Drop rows with NaN values

    # Create sequences and targets
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Prepare the last sequence for prediction
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    # Split data into train and test sets
    X, y = zip(*sequence_data)
    X, y = np.array(X), np.array(y)
    train_samples = int((1 - test_size) * len(X))
    result["X_train"], result["y_train"] = X[:train_samples], y[:train_samples]
    result["X_test"], result["y_test"] = X[train_samples:], y[train_samples:]

    # Retrieve test dates and construct test dataframe
    test_dates = result["X_test"][:, -1, -1]
    if not pd.api.types.is_datetime64_any_dtype(result["df"].index):
        result["df"].index = pd.to_datetime(result["df"].index)  # Convert index to datetime if needed
    test_dates = pd.to_datetime(test_dates)  # Convert test dates to datetime
    valid_dates = result["df"].index.intersection(test_dates)  # Only use valid dates in the dataframe index

    # Create test dataframe with valid dates
    result["test_df"] = result["df"].loc[valid_dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

    return result

# Function to create a model
def create_model(n_steps, n_features, units, cell, n_layers, dropout, activation, loss, optimizer):
    model = Sequential()
    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        model.add(cell(units, return_sequences=return_sequences, input_shape=(n_steps, n_features)))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation))
    model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    return model

# Function to predict k days into the future
def multi_step_predict(model, data, k):
    current_input = data['last_sequence']
    feature_count = current_input.shape[1]
    predictions = []
    for _ in range(k):
        current_input_reshaped = current_input.reshape(1, -1, feature_count)
        next_prediction = model.predict(current_input_reshaped)[0, 0]
        predictions.append(next_prediction)
        new_feature_row = [next_prediction] * feature_count
        current_input = np.append(current_input[1:], [new_feature_row], axis=0)
    return np.array(predictions)

# Function to plot candlestick chart
def candlestick_chart(df, n, prediction, future):
    df_resampled = df.resample(f'{n}D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna() if n > 1 else df.copy()
    last_date = df_resampled.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future + 1)]
    fig = go.Figure(data=[go.Candlestick(x=df_resampled.index, open=df_resampled['Open'], high=df_resampled['High'],
                                         low=df_resampled['Low'], close=df_resampled['Close'], 
                                         increasing_line_color='green', decreasing_line_color='red', name='Candlestick')])
    fig.add_trace(go.Scatter(x=future_dates, y=prediction[:future], mode='lines+markers',
                             line=dict(color='orange', width=2), marker=dict(size=8), name=f'Next {future} Day Prediction'))
    fig.update_layout(title=f'Stock Price Candlestick Chart ({n} Trading Day(s) per Candle)', xaxis_title='Date',
                      yaxis_title='Price', template='plotly_dark', width=1200, height=600)
    fig.show()

# Function to plot boxplot chart
def boxplot_chart(df, n, step=5):
    rolling_windows = [df['Close'][i:i + n].values for i in range(0, len(df) - n + 1)]
    plt.figure(figsize=(12, 6))
    plt.boxplot(rolling_windows, showfliers=False)
    plt.xticks(ticks=np.arange(1, len(rolling_windows) + 1, step=step), labels=np.arange(1, len(rolling_windows) + 1, step=step))
    plt.title(f'Boxplot of {COMPANY} Stock Prices Over a Moving Window of {n} Trading Day(s)')
    plt.xlabel('Rolling Window Index')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

# Function to test different LSTM configurations
def test_lstm_configurations(dataset, batch_sizes, epoch_sizes, lengths, units=1024, activation='linear'):
    LSTM_Test_Accuracy_Data = pd.DataFrame(columns=['batch_size', 'epoch_size', 'length', 'Test Accuracy'])
    n_features = dataset.shape[1]  # Update: Get the number of features from the dataset

    for x in itertools.product(batch_sizes, epoch_sizes, lengths):
        print(f"Testing configuration: batch_size={x[0]}, epochs={x[1]}, length={x[2]}")  # Debug statement

        # Prepare data generator
        generator = TimeseriesGenerator(dataset, dataset, batch_size=x[0], length=x[2])
        
        # Create LSTM model with specified units and activation
        lstm_model = Sequential()
        lstm_model.add(LSTM(1024, activation='linear', input_shape=(x[2], n_features)))  # Custom units and activation
        lstm_model.add(Dense(1))  # Output layer
        lstm_model.compile(optimizer='adam', loss='mse')

        # Add early stopping callback to prevent overfitting and excessive training time
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train the model
        try:
            lstm_model.fit(generator, epochs=x[1], verbose=1, callbacks=[early_stopping])  # Use early stopping
        except KeyboardInterrupt:
            print("Training interrupted manually")
            break  # Allow manual interruption

        # Make predictions
        lstm_predictions_scaled = []
        batch = dataset[-x[2]:]
        current_batch = batch.reshape((1, x[2], n_features))  # Update: Reshape correctly

        # Predict on test data
        for i in range(len(dataset) - x[2]):
            lstm_pred = lstm_model.predict(current_batch)[0]
            lstm_predictions_scaled.append(lstm_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[lstm_pred]], axis=1)  # Update: Correct shape

        # Calculate accuracy
        predictions_test = pd.DataFrame(lstm_predictions_scaled)
        errors_test = abs(predictions_test.iloc[:, 0] - dataset[x[2]:].reshape(-1))
        mape_test = 100 * (errors_test / dataset[x[2]:].reshape(-1))
        accuracy_test = 100 - np.mean(mape_test)

        # Store results
        LSTM_Test_Accuracy_Data_One = pd.DataFrame(index=range(1), columns=['batch_size', 'epoch_size', 'length', 'Test Accuracy'])
        LSTM_Test_Accuracy_Data_One.loc[0, 'batch_size'] = x[0]
        LSTM_Test_Accuracy_Data_One.loc[0, 'epoch_size'] = x[1]
        LSTM_Test_Accuracy_Data_One.loc[0, 'length'] = x[2]
        LSTM_Test_Accuracy_Data_One.loc[0, 'Test Accuracy'] = accuracy_test
        
        # Use pd.concat() to append the data frame
        LSTM_Test_Accuracy_Data = pd.concat([LSTM_Test_Accuracy_Data, LSTM_Test_Accuracy_Data_One], ignore_index=True)
        print(f"Configuration complete. Accuracy: {accuracy_test:.2f}%\n")  # Debug statement

    return LSTM_Test_Accuracy_Data


# Define constants
COMPANY = 'AAPL'
SCALE = True
FUTURE = 15
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["Close", "Volume", "Open", "High", "Low"]
FEATURE_COLUMNS1 = ["Close"]
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
date_now = time.strftime("%Y-%m-%d")

# Load and process data
data = load_and_process_data(COMPANY, N_STEPS, SCALE, FUTURE, TEST_SIZE, FEATURE_COLUMNS)

# Model file path
model_dir = 'model'
model_file = f'{model_dir}/{COMPANY}-{N_STEPS}-{UNITS}-{CELL.__name__}-{N_LAYERS}-{DROPOUT}-{LOSS}-{OPTIMIZER}-{EPOCHS}-{BATCH}-{ACTIVATION}-{len(FEATURE_COLUMNS)}_model.keras'

# Create and/or load model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if os.path.isfile(model_file):
    model = load_model(model_file)
else:
    model = create_model(N_STEPS, len(FEATURE_COLUMNS), UNITS, CELL, N_LAYERS, DROPOUT, ACTIVATION, LOSS, OPTIMIZER)
    model.fit(data["X_train"], data["y_train"], epochs=EPOCHS, batch_size=BATCH, 
            validation_data=(data["X_test"], data["y_test"]), 
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ])
    model.save(model_file)

# Predict using the test data
X_test = data["X_test"]
y_test = data["y_test"]
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values if scaling was applied
if SCALE:
    y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))

# Prepare the final dataframe
test_df = data["test_df"]
test_df[f"close_{FUTURE}"] = y_pred
test_df[f"true_close_{FUTURE}"] = y_test
test_df.sort_index(inplace=True)
final_df = test_df

# Test different configurations
dataset = pd.DataFrame(y_test).values
batch_sizes = [1, 2, 4]
epoch_sizes = [5, 7, 10]
lengths = [7, 30, 120]
lstm_accuracy_data = test_lstm_configurations(dataset, batch_sizes, epoch_sizes, lengths)
print(lstm_accuracy_data)

# Predict k days into the future
k_days_predicted_price = multi_step_predict(model, data, FUTURE)
price = data["column_scaler"]["Close"].inverse_transform(k_days_predicted_price.reshape(-1, 1))
print(f"Predicted price after {FUTURE} days: {price}")

plt.figure(figsize=(30,10))
plt.plot(final_df[f'true_close_{FUTURE}'], c='b')
plt.plot(final_df[f'close_{FUTURE}'], c='r')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend(["Actual Price", "Predicted Price"])
plt.show()

# Optional: Plot candlestick and boxplot charts
# candlestick_chart(data['df'], 1, y_pred, FUTURE)
# boxplot_chart(data['df'], 5)
