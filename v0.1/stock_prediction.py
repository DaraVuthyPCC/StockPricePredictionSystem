## Import necessary packages
import os
import time
import requests
import json
import csv
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from collections import deque
from tensorflow.keras.regularizers import l2
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    """
    Desc: This function will load data of 'company' from yahoo finance and then process the data by removing NaNs,
        scaling, based on multiple features, splitting the data into train set and test set and making a sequence based
        on n_steps and lookup_step for future prediction
    Parameters:
        company: the name of the company I choose, 'AAPL' 
        n_steps: the number of days to look back to based the prediction
        scale: scaling boolean
        lookup_step: the number of days into the future to predict
        test_size: the ratio of splitting the data into train and test, 0.2
        feature_columns: the features that we choose such as close, open, high, low
    """
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
    
    if df.index.name == 'Date' or 'Date' in df.index.names:
        df = df.reset_index()
        
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    
    # Loading and processing Sentiment Data     
    sentiment_data = pd.read_csv('./data/gdelt_aapl_sentiment_finbert.csv')
    sentiment_data['published_date'] = pd.to_datetime(sentiment_data['published_date'], errors='coerce')
    sentiment_data = sentiment_data.drop_duplicates(subset='published_date')
    full_date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    sentiment_data = sentiment_data.set_index('published_date').reindex(full_date_range).rename_axis('Date').reset_index()
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'], errors='coerce').dt.date
    sentiment_data['finbert_sentiment'] = sentiment_data['finbert_sentiment'].fillna('Neutral')
    sentiment_data['title'] = sentiment_data['title'].fillna("No Title")
    sentiment_data['url'] = sentiment_data['url'].fillna("No URL")
    sentiment_data['domain'] = sentiment_data['domain'].fillna("Unknown Domain")
    sentiment_data = sentiment_data.rename(columns={'finbert_sentiment': 'Sentiment Score'})
    df = pd.merge(df, sentiment_data, how='left', left_on='Date', right_on='Date')

    # Validate feature columns
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

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
    result['price_value'] = df['Close'].values

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
    """
    Desc: This function will create a model based on the n_steps and n_features that are training based on the units, cell
        n_layers, dropout, activation, loss, optimizer
    Parameters:
        n_steps: the number of days to look back to based the prediction, 50
        n_features: the number of features that I use to load the data and create the model, 5
        units: the number of neurons in each cell layer, 1024
        cell: the type of recurrent cell, LSTM
        n_layers: the number of recurrent layer, 4
        dropout: the dropout rate of input units each update during training. 0.5
        activation: the activation function, linear
        loss: the loss function that measure the difference between predicted and actual values, mean_sqaured_error
        optimizer: optimization algorithm, adam
    """
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
    """
    Desc: predict the closing prices for the next k days using a trained model.
    Parameters:
        model: A trained machine learning model capable of making predictions.
        data: A dictionary containing the 'last_sequence' or the initial input sequence for prediction.
        k: The number of future steps (days) to predict.
    """
    # Extract the initial input from the data dictionary
    current_input = data['last_sequence']
    # Get the number of features from the input
    feature_count = current_input.shape[1]
    # Initialize predictions list
    predictions = []
    for _ in range(k):
        # Reshape the input to match the model's expected input shape (1, n_steps, n_features)
        current_input_reshaped = current_input.reshape(1, -1, feature_count)
        # Make the prediction using the model
        next_prediction = model.predict(current_input_reshaped)[0, 0]  # Assuming model outputs a 2D array [[prediction]]
        # Append the prediction to the predictions list
        predictions.append(next_prediction)
        # Update the input by removing the first step and adding the new prediction
        new_feature_row = [next_prediction] * feature_count  # Extend prediction with same values for all features
        current_input = np.append(current_input[1:], [new_feature_row], axis=0)
    return np.array(predictions)

# Adjusting the prediction based on the sentiment
def adjust_prediction_with_sentiment(prediction, sentiment_score):
    """
    Desc: adjusting the current prediction with the sentiment getting from news
    Parameters:
        prediction: the prediction that the model have made
        sentiment_score: the sentiment score for news on that prediction day
    """
    if sentiment_score == 'Positive':
        return prediction * 1.05  # Increase prediction by 5% for positive sentiment
    elif sentiment_score == 'Negative':
        return prediction * 0.95  # Decrease prediction by 5% for negative sentiment
    else:
        return prediction  # Neutral sentiment: keep the prediction as is

# Arima/Sarimax ensemble with LSTM
def arima_and_sarimax_lstm_ensemble(data, lstm_model, k_days, scale, feature_columns):
    """
    Desc: ensemble between arima-LSTM and sarimax-LSTM to improve accuracy
    Parameters:
        data: the testing data that was split in the load_and_process_data function
        lstm_model: the lstm model that I have trained early on
        k_days: future days to predict
        scale: scaling the value of each feature
        feature_columns: the features that used for train and test
    """
    # Extract the close prices from the dataset
    close_prices = data["df"]['Close'].values
    # exog for SARIMAX multivariate method
    exog_columns = [col for col in feature_columns if col != 'Close']
    exog = data["df"][exog_columns].values 

    # Get the latest date in the data
    last_date = data["df"]['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=k_days).to_pydatetime()  # Future k_days dates
    
    # Train ARIMA and SARIMAX model on the close prices
    arima_model = ARIMA(close_prices, order=(5,1,0))
    arima_model_fit = arima_model.fit()

    sarimax_model = SARIMAX(close_prices, exog=exog, order=(4,1,1), seasonal_order=(1,1,1,6))
    sarimax_model_fit = sarimax_model.fit()

    # Get ARIMA and SARIMAX predictions over the entire test set and next k_days
    arima_pred = arima_model_fit.predict(start=len(data["X_train"]), end=len(data["df"])-1)
    arima_future_pred = arima_model_fit.forecast(steps=k_days)

    exog_future = exog[-k_days:]
    sarimax_pred = sarimax_model_fit.predict(start=len(data["X_train"]), end=len(data["df"])-1)
    sarimax_future_pred = sarimax_model_fit.forecast(steps=k_days, exog=exog_future)
    
    # Get the LSTM predictions over the entire test set and next k_days
    lstm_pred = lstm_model.predict(data["X_test"])
    if scale:
        lstm_pred = data["column_scaler"]["Close"].inverse_transform(lstm_pred)

    lstm_future_pred = multi_step_predict(lstm_model, data, k_days)
    if scale:
        lstm_future_pred = data["column_scaler"]["Close"].inverse_transform(lstm_future_pred.reshape(-1, 1))    
        
    # Ensure the shapes are the same by trimming or interpolating the longer array
    min_length = min(len(arima_pred), len(lstm_pred))
    arima_pred = arima_pred[-min_length:]
    sarimax_pred = sarimax_pred[-min_length:]
    lstm_pred = lstm_pred[-min_length:]
        
    # Combine ARIMA/SARIMAX and LSTM predictions
    arima_ensemble_pred = (arima_pred + lstm_pred.flatten()) / 2
    sarimax_ensemble_pred = (sarimax_pred + lstm_pred.flatten()) / 2

    # Apply sentiment adjustment
    for i in range(min_length):
        sentiment = data["df"].iloc[i]["Sentiment Score"]
        arima_ensemble_pred[i] = adjust_prediction_with_sentiment(arima_ensemble_pred[i], sentiment)
        sarimax_ensemble_pred[i] = adjust_prediction_with_sentiment(sarimax_ensemble_pred[i], sentiment)
    
    # Get the last available sentiment
    last_known_sentiment = data["df"]["Sentiment Score"].iloc[-1]

    # Future predictions for next k_days and let the k_days sentiment the same as the last sentiment available
    arima_future_ensemble_pred = [adjust_prediction_with_sentiment(pred, last_known_sentiment) for pred in arima_future_pred]
    sarimax_future_ensemble_pred = [adjust_prediction_with_sentiment(pred, last_known_sentiment) for pred in sarimax_future_pred]

    arima_future_ensemble_pred = (arima_future_pred + lstm_future_pred.flatten()) / 2
    sarimax_future_ensemble_pred = (sarimax_future_pred + lstm_future_pred.flatten()) / 2
    
    # Future predictions for next k_days and let the k_days sentiment the same as the last sentiment available
    arima_future_ensemble_pred = [adjust_prediction_with_sentiment(pred, last_known_sentiment) for pred in arima_future_pred]
    sarimax_future_ensemble_pred = [adjust_prediction_with_sentiment(pred, last_known_sentiment) for pred in sarimax_future_pred]
    
    # Get the actual close prices for the test set
    y_test = data["y_test"][-min_length:]
    if scale:
        y_test = data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0))

    actual_future_prices = data["df"]["Close"][-k_days:].values

    results = {
        "future_dates": future_dates,
        "lstm_pred": lstm_pred,
        "arima_ensemble_pred": arima_ensemble_pred,
        "sarimax_ensemble_pred": sarimax_ensemble_pred,
        "y_test": y_test,
        "lstm_future_pred": lstm_future_pred,
        "arima_future_ensemble_pred": arima_future_ensemble_pred,
        "sarimax_future_ensemble_pred": sarimax_future_ensemble_pred,
        "actual_future_prices": actual_future_prices
    }
    
    return results

# Calculate error metrics for LSTM, ARIMA, and SARIMAX ensembles
def evaluation(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

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
# model_file = f'{model_dir}/{COMPANY}-{N_STEPS}-{UNITS}-{CELL.__name__}-{N_LAYERS}-{DROPOUT}-{LOSS}-{OPTIMIZER}-{EPOCHS}-{BATCH}-{ACTIVATION}-{len(FEATURE_COLUMNS)}_model.keras'
model_file = f'./model/AAPL-100-1024-LSTM-4-0.5-mean_squared_error-adam-100-128-linear-5_model.keras'
# model_file = '/kaggle/input/lstm/keras/default/1/AAPL-100-1024-LSTM-4-0.5-mean_squared_error-adam-100-128-linear-5_model.keras'

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

# # Predict using the test data
# X_test = data["X_test"]
# y_test = data["y_test"]
# y_pred = model.predict(X_test)

# # Inverse transform predictions and actual values if scaling was applied
# if SCALE:
#     y_test = np.squeeze(data["column_scaler"]["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
#     y_pred = np.squeeze(data["column_scaler"]["Close"].inverse_transform(y_pred))

# Prepare the final dataframe
# test_df = data["test_df"]
# test_df[f"close_{FUTURE}"] = y_pred
# test_df[f"true_close_{FUTURE}"] = y_test
# test_df.sort_index(inplace=True)
# final_df = test_df

# k_days_predicted_price = multi_step_predict(model, data, FUTURE)
# price = data["column_scaler"]["Close"].inverse_transform(k_days_predicted_price.reshape(-1, 1))

# Optional: Plot candlestick and boxplot charts
# candlestick_chart(data['df'], 1, y_pred, FUTURE)
# boxplot_chart(data['df'], 5)

# Predicting using ensemble
results = arima_and_sarimax_lstm_ensemble(data, model, FUTURE, SCALE, FEATURE_COLUMNS)

prediction_df = pd.DataFrame({
    'Date': results['future_dates'],
    'LSTM Price': results['lstm_future_pred'].flatten(),
    'ARIMA Price': results['arima_future_ensemble_pred'],
    'SARIMAX Price': results['sarimax_future_ensemble_pred'],
    'Actual Price': results['actual_future_prices']
})

print(prediction_df)

# Evaluation
lstm_eval = evaluation(prediction_df['Actual Price'], prediction_df['LSTM Price'])
arima_eval = evaluation(prediction_df['Actual Price'], prediction_df['ARIMA Price'])
sarimax_eval = evaluation(prediction_df['Actual Price'], prediction_df['SARIMAX Price'])

# Print the results
for model_name, eval in zip(['LSTM', 'ARIMA Ensemble', 'SARIMAX Ensemble'], [lstm_eval, arima_eval, sarimax_eval]):
    print(f"{model_name} Prediction Error Metrics:\nMAE: {eval[0]}\nMSE: {eval[1]}\nRMSE: {eval[2]}\n")

# Plot the full actual prices vs model predicted prices
plt.figure(figsize=(30, 10))
plt.plot(results['y_test'].flatten(), c='b', label='Actual Price')
plt.plot(results['arima_ensemble_pred'], c='g', label='ARIMA Ensemble Predicted Price')
plt.plot(results['sarimax_ensemble_pred'], c='y', label='SARIMAX Ensemble Predicted Price')
plt.plot(results['lstm_pred'], c='r', label='LSTM Predicted Price')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot the next k_days price
plt.figure(figsize=(10,5))
plt.plot(range(1, FUTURE + 1), results['actual_future_prices'].flatten(), c='b', label='Actual Future Price')
plt.plot(range(1, FUTURE + 1), results['arima_future_ensemble_pred'], c='g', label='ARIMA Ensemble Future Predicted Price')
plt.plot(range(1, FUTURE + 1), results['sarimax_future_ensemble_pred'], c='y', label='SARIMAX Ensemble Future Predicted Price')
plt.plot(range(1, FUTURE + 1), results['lstm_future_pred'], c='r', label='LSTM Future Predicted Price')
plt.xlabel(f"Days into Future")
plt.ylabel("Price")
plt.legend()
plt.show()

# ensemble arima with randomforest
def arima_rf_ensemble(data, arima_order, n_lags):
    """
    Desc: Combines ARIMA and Random Forest models using averaging.
    Parameters:
        data: DataFrame containing the time series data with 'Close' prices.
        arima_order: Tuple (p, d, q) for ARIMA model.
        n_lags: Number of lagged features for Random Forest.
    """
    # training the arima     
    close_prices = data["df"]['Close'].values
    arima_model = ARIMA(close_prices, order=arima_order)
    arima_model_fit = arima_model.fit()

    # Get ARIMA predictions
    arima_predictions = arima_model_fit.forecast(steps=len(data["df"]['Close']))

    # creating lag feature for randomforest
    def create_lagged_features(df, lag=n_lags):
        df_lagged = pd.DataFrame()
        df_lagged['Close'] = df["df"]['Close']
        for i in range(1, lag + 1):
            df_lagged[f'Close_lag_{i}'] = df["df"]['Close'].shift(i)
        df_lagged.dropna(inplace=True)
        return df_lagged

    lagged_df = create_lagged_features(data)

    # Split the data into training and test sets
    train_size = int(0.8 * len(lagged_df))
    train, test = lagged_df[:train_size], lagged_df[train_size:]

    # Separate features and target for Random Forest
    X_train, y_train = train.drop('Close', axis=1), train['Close']
    X_test, y_test = test.drop('Close', axis=1), test['Close']

    # training randomforest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get Random Forest predictions
    rf_predictions = rf_model.predict(X_test)

    # ensemble arima with randomforest by simple averaging
    arima_predictions = arima_predictions[-len(y_test):]
    ensemble_predictions = (arima_predictions + rf_predictions) / 2

    # Evaluate the ensemble
    ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)
    ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
    ensemble_rmse = np.sqrt(ensemble_mse)

    # Print the results
    print("ARIMA + Random Forest Ensemble Prediction Error Metrics:")
    print(f"MAE: {ensemble_mae}")
    print(f"MSE: {ensemble_mse}")
    print(f"RMSE: {ensemble_rmse}")

    return ensemble_predictions, y_test

pred, ytest = arima_rf_ensemble(data, (5, 1, 0), 3)

print(pred)