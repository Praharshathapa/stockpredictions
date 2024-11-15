from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import logging
import os
import pandas as pd


# Set up logging for error tracking
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Fetch stock data for multiple tickers (last 6 months)
def fetch_stock_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, period="6mo")[["Close"]]
            # Ensure enough data is available for the model (at least 60 days)
            if len(stock_data) >= 60:
                stock_data.dropna(inplace=True)
                data[ticker] = stock_data
            else:
                logging.warning(f"Not enough data for {ticker}")
        except Exception as e:
            logging.error(f"Failed to fetch data for {ticker}: {str(e)}")
    return data

# Determine buy/sell/hold action based on short-term and long-term predictions
def determine_buy_sell_action(current_price, short_term_pred, long_term_pred, short_term_threshold=0.02, long_term_threshold=0.05):
    # Ensure that inputs are scalar values, extracting a single value if they are Series
    current_price = float(current_price) if isinstance(current_price, (np.ndarray, pd.Series)) else current_price
    short_term_pred = float(short_term_pred) if isinstance(short_term_pred, (np.ndarray, pd.Series)) else short_term_pred
    long_term_pred = float(long_term_pred) if isinstance(long_term_pred, (np.ndarray, pd.Series)) else long_term_pred

    short_term_change = (short_term_pred - current_price) / current_price
    long_term_change = (long_term_pred - current_price) / current_price

    short_term_decision = "buy" if short_term_change > short_term_threshold else "sell" if short_term_change < -short_term_threshold else "hold"
    long_term_decision = "buy" if long_term_change > long_term_threshold else "sell" if long_term_change < -long_term_threshold else "hold"

    if short_term_decision == "buy" and long_term_decision == "buy":
        return "Strong Buy"
    elif short_term_decision == "sell" and long_term_decision == "sell":
        return "Strong Sell"
    elif short_term_decision == "buy" or long_term_decision == "buy":
        return "Buy"
    elif short_term_decision == "sell" or long_term_decision == "sell":
        return "Sell"
    else:
        return "Hold"


# Prepare the data for LSTM model
def prepare_data_for_lstm(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])  # Create 60-day window of data
        y.append(scaled_data[i, 0])  # Predict the next day's closing price

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    return X, y, scaler

# Build and compile the LSTM model
def build_lstm_model(X_train, y_train, model_name):
    model_path = f"{model_name}_lstm_model.pkl"
    
    # Check if the model exists
    if os.path.exists(model_path):
        # Load model if already trained and saved
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    # Otherwise, build and train the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predict the closing price

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Save the model using pickle
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tickers = request.form["tickers"].split(",")
        tickers = [ticker.strip().upper() for ticker in tickers]

        try:
            data = fetch_stock_data(tickers)
            predictions = {}
            decisions = {}

            for ticker in tickers:
                if ticker in data:
                    stock_data = data[ticker]

                    # Prepare data for LSTM
                    X, y, scaler = prepare_data_for_lstm(stock_data)

                    # Split the data into training and testing sets
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    # Build and train or load the LSTM model
                    model = build_lstm_model(X_train, y_train, ticker)

                    # Predict short-term and long-term prices
                    short_term_pred = model.predict(X_test[-1:])[0][0]
                    long_term_pred = model.predict(X_test[:1])[0][0]

                    # Scale predictions back to original prices
                    current_price = stock_data["Close"].iloc[-1]
                    short_term_pred = scaler.inverse_transform([[short_term_pred]])[0][0]
                    long_term_pred = scaler.inverse_transform([[long_term_pred]])[0][0]

                    # Store the predictions
                    predictions[ticker] = {
                        "current_price": current_price,
                        "short_term_pred": short_term_pred,
                        "long_term_pred": long_term_pred
                    }

                    # Determine buy/sell/hold action
                    decisions[ticker] = determine_buy_sell_action(current_price, short_term_pred, long_term_pred)

            return render_template("index.html", predictions=predictions, decisions=decisions, tickers=tickers)

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
