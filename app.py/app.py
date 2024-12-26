from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

# Updated list of stock symbols (Companies)
companies = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'WMT', 'NFLX', 'ORCL', 'META', 'MCD', 'BAC']

# Function to fetch stock data
def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1y")  # Get 1 year of stock data
    return data

# Feature Engineering for the data
def prepare_data(data):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date']).map(lambda x: x.timestamp())
    features = data[['Date']]
    target = data['Close']
    return features, target

# Train the model and save it to disk
def train_and_save_model(stock_symbol, features, target):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model and scaler with the stock symbol in the filename
    pickle.dump(model, open(f'model_{stock_symbol}.pkl', 'wb'))
    pickle.dump(scaler, open(f'scaler_{stock_symbol}.pkl', 'wb'))

    return model, scaler

# Predict the next day's stock price
def predict_next_day(model, scaler, last_date):
    last_date_scaled = scaler.transform([[last_date]])
    prediction = model.predict(last_date_scaled)
    return prediction[0]

# Load model and scaler for a specific stock symbol if they exist, otherwise train a new model
def load_model(stock_symbol):
    model_path = f'model_{stock_symbol}.pkl'
    scaler_path = f'scaler_{stock_symbol}.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        return model, scaler
    else:
        return None, None

# Function to calculate stock growth (percentage change over 1 year)
def calculate_growth(data):
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    growth = ((end_price - start_price) / start_price) * 100
    return growth, start_price, end_price

# Flask route for the main page (UI)
@app.route('/')
def index():
    return render_template('index.html', companies=companies)

# Flask route for prediction logic
@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('symbol')

    if not stock_symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    # Fetch stock data for the selected company
    data = fetch_stock_data(stock_symbol)
    features, target = prepare_data(data)

    # Try to load the model specific to this stock
    model, scaler = load_model(stock_symbol)

    # If no model exists, train a new one
    if model is None or scaler is None:
        model, scaler = train_and_save_model(stock_symbol, features, target)

    # Get the last date for prediction
    last_date = features['Date'].iloc[-1]

    # Predict the next day's stock price
    next_day_prediction = predict_next_day(model, scaler, last_date)

    # Calculate stock growth over the past year
    growth, start_price, end_price = calculate_growth(data)

    # Suggest Buy/Sell/Hold
    suggestion = "Hold"
    if growth > 10:
        suggestion = "Buy"
    elif growth < 0:
        suggestion = "Sell"

    return jsonify({
        'prediction': next_day_prediction,
        'growth': growth,
        'start_price': start_price,
        'end_price': end_price,
        'suggestion': suggestion
    })

if __name__ == '__main__':
    app.run(debug=True)
