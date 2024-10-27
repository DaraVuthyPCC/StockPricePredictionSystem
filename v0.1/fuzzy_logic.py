from sklearn.metrics import mean_squared_error, mean_absolute_error
import skfuzzy as fuzz
import numpy as np
import pandas as pd
import yfinance as yf

from collections import deque
from datetime import timedelta, datetime as dt
from sklearn.preprocessing import MinMaxScaler


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

    # Validate feature columns
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    result = {'df': df.copy()}
    # Scale the data if required
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[column])
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

def fuzzify_data(data):
    # flatten the 'Close' column into 1D dataframe
    data['df'][('Close', 'AAPL')] = pd.to_numeric(data['df'][('Close', 'AAPL')], errors='coerce').values.flatten()
    
    # Drop any rows where 'Close' is NaN
    data['df'] = data['df'].dropna(subset=[('Close', 'AAPL')])
    
    # Define fuzzy membership functions
    low = fuzz.trapmf(data['df'][('Close', 'AAPL')], [min(data['df'][('Close', 'AAPL')]), min(data['df'][('Close', 'AAPL')]), 
                                                      0.3 * max(data['df'][('Close', 'AAPL')]), 0.5 * max(data['df'][('Close', 'AAPL')])])
    medium = fuzz.trimf(data['df'][('Close', 'AAPL')], [0.3 * max(data['df'][('Close', 'AAPL')]), 0.5 * max(data['df'][('Close', 'AAPL')]), 
                                                        0.7 * max(data['df'][('Close', 'AAPL')])])
    high = fuzz.trapmf(data['df'][('Close', 'AAPL')], [0.5 * max(data['df'][('Close', 'AAPL')]), 0.7 * max(data['df'][('Close', 'AAPL')]), 
                                                       max(data['df'][('Close', 'AAPL')]), max(data['df'][('Close', 'AAPL')])])
    
    # Apply fuzzy membership to the stock data
    data['df']['Low'] = fuzz.interp_membership(data['df'][('Close', 'AAPL')], low, data['df'][('Close', 'AAPL')])
    data['df']['Medium'] = fuzz.interp_membership(data['df'][('Close', 'AAPL')], medium, data['df'][('Close', 'AAPL')])
    data['df']['High'] = fuzz.interp_membership(data['df'][('Close', 'AAPL')], high, data['df'][('Close', 'AAPL')])

    return data

def fuzzy_regression_model(fuzzified_data):
    # Implement the fuzzy regression formula based on triangular membership functions
    lmf = fuzzified_data['df']['Low']
    umf = fuzzified_data['df']['High']
    
    # Compute fuzzy coefficients
    a = np.mean(lmf)
    b = np.mean(umf)
    
    # Predict stock prices based on fuzzy logic
    predicted = a * lmf + b * umf
    
    return predicted


COMPANY = 'AAPL'
SCALE = True
FUTURE = 15
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["Close", "Volume", "Open", "High", "Low"]
N_STEPS = 100

# Load and process data
data = load_and_process_data(COMPANY, N_STEPS, SCALE, FUTURE, TEST_SIZE, FEATURE_COLUMNS)

# Fuzzify the stock data
fuzzified_data = fuzzify_data(data)

# Apply fuzzy regression model
predicted_values = fuzzy_regression_model(fuzzified_data)

# Evaluate the model performance (using RMSE as in the paper)
actual_values = data['df']['Close']
mae = mean_absolute_error(actual_values, predicted_values)
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')