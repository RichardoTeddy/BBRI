import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, render_template, request
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json
import yfinance as yf

# Configure matplotlib
matplotlib.use('Agg')
plt.style.use('fivethirtyeight')

# Initialize Flask app
app = Flask(__name__)

# Global variables
MODEL_PATH = 'model.keras'
DATA_PATH = 'BBRI.JK.NEW.csv'
SEQUENCE_LENGTH = 100
MAX_PREDICTION_DAYS = 30
model = None


def load_model_safe():
    """Load the LSTM model with proper error handling"""
    global model
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(
                f"Error: Model file not found at {os.path.abspath(MODEL_PATH)}")
            return None

        # Check if model file is empty
        if os.path.getsize(MODEL_PATH) == 0:
            print("Error: Model file is empty")
            return None

        # Load the model
        print(f"Loading model from {os.path.abspath(MODEL_PATH)}")
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error')

        print("Model loaded successfully")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model path: {os.path.abspath(MODEL_PATH)}")
        return None

# Create model test function


def test_model():
    """Test if model can be loaded and make predictions"""
    try:
        # Load model
        test_model = load_model(MODEL_PATH, compile=False)
        test_model.compile(optimizer='adam', loss='mean_squared_error')

        # Create dummy data to test prediction
        dummy_data = np.random.random((1, SEQUENCE_LENGTH, 1))

        # Test prediction
        _ = test_model.predict(dummy_data)
        print("Model test successful!")
        return True
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False


# Initialize model when app starts
print("Initializing model...")
model = load_model_safe()

if model is None:
    print("""
    ERROR: Model could not be loaded. Please ensure:
    1. The model file 'model.keras' exists in the current directory
    2. The model file is not corrupted
    3. You have the correct version of TensorFlow installed
    """)
else:
    # Test the model
    if test_model():
        print("Model initialization complete and working!")
    else:
        print("Model loaded but failed testing!")


def save_history(history):
    """Save training history to JSON file"""
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open('model_history.json', 'w') as f:
        json.dump(history_dict, f)


def load_history():
    """Load training history from JSON file"""
    try:
        with open('model_history.json', 'r') as f:
            return json.load(f)
    except:
        return {'loss': [], 'val_loss': []}


def load_stock_data():
    """Load and preprocess stock data, download if needed"""
    try:
        # Try to load existing data
        if os.path.exists(DATA_PATH):
            # Read CSV with specific date parsing
            stock_data = pd.read_csv(DATA_PATH)
            stock_data['Date'] = pd.to_datetime(
                stock_data['Date'], format='mixed', utc=True)

            # Convert to local timezone and remove timezone info
            stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)

            # Check if data needs updating
            last_date = stock_data['Date'].max()
            today = pd.Timestamp.now()

            if last_date.date() < today.date() - timedelta(days=1):
                # Data needs updating
                print("Updating stock data from Yahoo Finance...")
                new_data = download_stock_data(
                    last_date + timedelta(days=1), today)

                if not new_data.empty:
                    # Ensure new data has consistent date format
                    new_data['Date'] = pd.to_datetime(
                        new_data['Date'], format='mixed', utc=True)
                    new_data['Date'] = new_data['Date'].dt.tz_localize(None)

                    # Combine old and new data
                    stock_data = pd.concat([stock_data, new_data])

                    # Remove duplicates if any
                    stock_data = stock_data.drop_duplicates(subset=['Date'])

                    # Sort by date
                    stock_data = stock_data.sort_values('Date')

                    # Save updated data
                    stock_data.to_csv(DATA_PATH, index=False,
                                      date_format='%Y-%m-%d')
                    print("Stock data updated successfully!")
        else:
            # File doesn't exist, download all data
            print("Downloading historical stock data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)  # 5 years of data
            stock_data = download_stock_data(start_date, end_date)

            if not stock_data.empty:
                # Ensure consistent date format
                stock_data['Date'] = pd.to_datetime(
                    stock_data['Date'], format='mixed', utc=True)
                stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)

                # Save new data
                stock_data.to_csv(DATA_PATH, index=False,
                                  date_format='%Y-%m-%d')
                print("Stock data downloaded and saved successfully!")
            else:
                raise ValueError("Failed to download stock data")

        # Set index after ensuring dates are properly formatted
        stock_data.set_index('Date', inplace=True)
        return stock_data

    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        raise


def download_stock_data(start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker("BBRI.JK")
        data = ticker.history(start=start_date, end=end_date)

        # Reset index to make Date a column and remove timezone info
        data = data.reset_index()
        data['Date'] = data['Date'].dt.tz_localize(None)

        # Rename columns to match existing format
        data = data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })

        # Select only needed columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        return data

    except Exception as e:
        print(f"Error downloading stock data: {str(e)}")
        return pd.DataFrame()


def prepare_prediction_data(stock_data, days):
    """Prepare data for prediction"""
    scaler = MinMaxScaler()

    # Reshape data to 2D array
    close_values = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_values)

    # Ensure we have enough data
    if len(scaled_data) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough data. Need at least {SEQUENCE_LENGTH} data points.")

    # Get the last SEQUENCE_LENGTH data points
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]

    # Reshape to match the expected input shape (samples, time steps, features)
    X_predict = last_sequence.reshape(1, SEQUENCE_LENGTH, 1)

    return X_predict, scaler


def generate_predictions(model, X_predict, scaler, days):
    """Generate price predictions"""
    predictions = []
    current_batch = X_predict.copy()

    for _ in range(days):
        # Make prediction
        current_pred = model.predict(current_batch, verbose=0)
        predictions.append(current_pred[0, 0])

        # Update the sequence by removing the first element and adding the prediction
        current_batch = current_batch.reshape(SEQUENCE_LENGTH, 1)
        current_batch = np.vstack([current_batch[1:], current_pred])
        current_batch = current_batch.reshape(1, SEQUENCE_LENGTH, 1)

    # Convert predictions to price values
    predictions = np.array(predictions).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predictions)

    return predicted_prices

# Flask routes


@app.route("/")
def index():
    return render_template("index.html", title="Beranda")


@app.route("/bbri")
def bbri():
    return render_template("bbri.html", title="BBRI")


@app.route("/lstm")
def lstm():
    return render_template("lstm.html", title="LSTM")


@app.route("/input")
def input():
    return render_template("input.html", title="Input")


@app.route("/hasil")
def hasil():
    try:
        # Load stock data
        stock_data = load_stock_data()

        # Generate Close Price History plot
        plt.figure(figsize=(15, 6))
        plt.plot(stock_data.index,
                 stock_data['Close'], color='blue', linewidth=2)
        plt.title('Historical Close Price BBRI')
        plt.xlabel('Date')
        plt.ylabel('Price (IDR)')
        plt.grid(True, alpha=0.3)

        # Save close price plot
        close_price_buffer = io.BytesIO()
        plt.savefig(close_price_buffer, format='png', bbox_inches='tight')
        close_price_buffer.seek(0)
        close_price_plot = base64.b64encode(
            close_price_buffer.getvalue()).decode()
        plt.close()

        # Calculate Moving Averages
        ma_100 = stock_data['Close'].rolling(window=100).mean()
        ma_365 = stock_data['Close'].rolling(window=365).mean()

        # Generate Moving Average plot
        plt.figure(figsize=(15, 6))
        plt.plot(stock_data.index, stock_data['Close'],
                 label='Close Price', color='blue', linewidth=2)
        plt.plot(stock_data.index, ma_100, label='MA-100',
                 color='green', linestyle='--', linewidth=2)
        plt.plot(stock_data.index, ma_365, label='MA-365',
                 color='red', linestyle='--', linewidth=2)
        plt.title('Close Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (IDR)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save MA plot
        ma_buffer = io.BytesIO()
        plt.savefig(ma_buffer, format='png', bbox_inches='tight')
        ma_buffer.seek(0)
        ma_plot = base64.b64encode(ma_buffer.getvalue()).decode()
        plt.close()

        return render_template(
            "hasil.html",
            title="Hasil Analisis",
            close_price_plot=f"data:image/png;base64,{close_price_plot}",
            ma_plot=f"data:image/png;base64,{ma_plot}"
        )

    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return render_template(
            "hasil.html",
            title="Hasil Analisis",
            error="Terjadi kesalahan dalam menghasilkan visualisasi"
        )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests"""
    try:
        # Check if model is loaded
        if model is None:
            raise ValueError(
                "Model belum dimuat. Pastikan file model.keras tersedia dan valid.")

        # Get and validate inputs
        start_date = pd.to_datetime(request.form.get("start_date"))
        end_date = pd.to_datetime(request.form.get("end_date"))
        train_split = float(request.form.get("train_split", 80)) / 100
        days = int(request.form.get("days-predict", 0))

        if not all([start_date, end_date, train_split, days]):
            return render_template(
                "input.html",
                error="Semua parameter harus diisi",
                title="Input"
            )

        # Load and filter data
        stock_data = load_stock_data()

        # Convert index to datetime if it's not already
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)

        # Filter data between start and end dates
        mask = (stock_data.index >= start_date) & (
            stock_data.index <= end_date)
        filtered_data = stock_data[mask].copy()

        if len(filtered_data) < SEQUENCE_LENGTH:
            return render_template(
                "input.html",
                error=f"Data tidak cukup. Minimal {SEQUENCE_LENGTH} hari data diperlukan.",
                title="Input"
            )

        # Split data for training evaluation
        train_size = int(len(filtered_data) * train_split)
        train_data = filtered_data[:train_size]
        test_data = filtered_data[train_size:]

        # Prepare prediction data and generate predictions
        X_predict, scaler = prepare_prediction_data(filtered_data, days)
        predicted_prices = generate_predictions(model, X_predict, scaler, days)

        # Generate future dates
        last_date = filtered_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )

        # Format prediction data
        prediction_data = [
            {'date': date.strftime('%Y-%m-%d'), 'price': float(price[0])}
            for date, price in zip(future_dates, predicted_prices)
        ]

        # Calculate metrics and generate plots
        mape = calculate_mape(model, test_data)
        plots = {
            'trend': generate_trend_plot(filtered_data, future_dates, predicted_prices),
            'validation': generate_validation_plot(model, filtered_data)
        }

        # Generate AI recommendation
        ai_recommendation = analyze_prediction_trend(
            filtered_data, prediction_data, mape)

        return render_template(
            "prediksi.html",
            title="Hasil Prediksi",
            plots=plots,
            prediction_data=prediction_data,
            mape=mape,
            ai_recommendation=ai_recommendation,
            train_split=train_split * 100
        )

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return render_template(
            "error.html",
            error=f"Terjadi kesalahan: {str(e)}",
            title="Error"
        )


def generate_trend_plot(stock_data, future_dates, future_prices):
    """Generate historical and future trend plot"""
    plt.figure(figsize=(12, 6))

    # Plot historical data (last 100 days)
    historical_dates = stock_data.index[-100:]
    historical_prices = stock_data['Close'].iloc[-100:]
    plt.plot(historical_dates, historical_prices,
             color='blue', label='Data Historis')

    # Plot future predictions
    plt.plot(future_dates, future_prices,
             color='orange', label='Prediksi')

    plt.title('Trend Historis dan Prediksi Harga')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (IDR)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f'<img src="data:image/png;base64,{plot_url}">'


def generate_validation_plot(model, stock_data):
    """Generate validation plot comparing actual vs predicted values"""
    if len(stock_data) < SEQUENCE_LENGTH:
        return ""

    plt.figure(figsize=(12, 6))

    # Prepare data
    scaler = MinMaxScaler()
    close_values = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_values)

    # Generate sequences
    X = []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
    X = np.array(X)
    X = X.reshape(X.shape[0], SEQUENCE_LENGTH, 1)

    # Generate predictions
    predictions = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    # Prepare actual values for plotting
    actual_dates = stock_data.index[SEQUENCE_LENGTH:]
    actual_prices = stock_data['Close'].iloc[SEQUENCE_LENGTH:]

    # Plot
    plt.plot(actual_dates, actual_prices, label='Aktual', color='blue')
    plt.plot(actual_dates, predictions, label='Prediksi', color='orange')

    plt.title('Perbandingan Harga Aktual vs Prediksi')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (IDR)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f'<img src="data:image/png;base64,{plot_url}">'


def calculate_mape(model, stock_data):
    """Calculate Mean Absolute Percentage Error"""
    if len(stock_data) < SEQUENCE_LENGTH:
        return 0

    scaler = MinMaxScaler()
    close_values = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_values)

    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        sequence = scaled_data[i:(i + SEQUENCE_LENGTH)]
        target = scaled_data[i + SEQUENCE_LENGTH]
        X.append(sequence)
        y.append(target)

    X = np.array(X)
    X = X.reshape(X.shape[0], SEQUENCE_LENGTH, 1)
    y = np.array(y)

    predictions = model.predict(X, verbose=0)

    # Convert predictions back to original scale
    predictions_original = scaler.inverse_transform(predictions)
    y_original = scaler.inverse_transform(y)

    # Calculate MAPE
    mape = np.mean(
        np.abs((y_original - predictions_original) / y_original)) * 100
    return float(mape)


def analyze_prediction_trend(historical_data, prediction_data, mape):
    """Analyze prediction trend and generate AI recommendation"""

    # Get the last 5 historical prices and predicted prices
    last_actual = historical_data['Close'].iloc[-5:].values
    predicted_prices = [p['price'] for p in prediction_data[:5]]

    # Calculate trend percentages
    hist_trend = ((last_actual[-1] - last_actual[0]) / last_actual[0]) * 100
    pred_trend = (
        (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]) * 100

    # Initialize recommendation
    recommendation = {
        'action': '',
        'confidence': '',
        'risk_level': '',
        'reasoning': []
    }

    # Determine action based on trends and MAPE
    if pred_trend > 2 and mape < 5:
        recommendation['action'] = 'BELI'
        confidence = 'TINGGI'
    elif pred_trend < -2 and mape < 5:
        recommendation['action'] = 'JUAL'
        confidence = 'TINGGI'
    elif abs(pred_trend) <= 2:
        recommendation['action'] = 'TAHAN'
        confidence = 'SEDANG'
    else:
        recommendation['action'] = 'TAHAN'
        confidence = 'RENDAH'

    # Determine risk level based on MAPE
    if mape < 3:
        risk_level = 'RENDAH'
    elif mape < 7:
        risk_level = 'SEDANG'
    else:
        risk_level = 'TINGGI'

    # Add reasoning
    recommendation['confidence'] = confidence
    recommendation['risk_level'] = risk_level

    # Generate reasoning based on analysis
    if recommendation['action'] == 'BELI':
        recommendation['reasoning'] = [
            f"Prediksi menunjukkan tren kenaikan {pred_trend:.1f}% dalam 5 hari ke depan",
            f"Tingkat akurasi model cukup baik dengan MAPE {mape:.2f}%",
            f"Tren historis menunjukkan pergerakan {hist_trend:.1f}% dalam 5 hari terakhir",
            "Momentum pasar positif berdasarkan analisis teknikal"
        ]
    elif recommendation['action'] == 'JUAL':
        recommendation['reasoning'] = [
            f"Prediksi menunjukkan tren penurunan {pred_trend:.1f}% dalam 5 hari ke depan",
            f"Tingkat akurasi model cukup baik dengan MAPE {mape:.2f}%",
            f"Tren historis menunjukkan pergerakan {hist_trend:.1f}% dalam 5 hari terakhir",
            "Indikasi teknikal menunjukkan momentum negatif"
        ]
    else:
        recommendation['reasoning'] = [
            f"Prediksi menunjukkan tren sideways {pred_trend:.1f}% dalam 5 hari ke depan",
            f"Tingkat akurasi model dengan MAPE {mape:.2f}%",
            f"Tren historis menunjukkan pergerakan {hist_trend:.1f}% dalam 5 hari terakhir",
            "Disarankan untuk menunggu sinyal yang lebih jelas"
        ]

    return recommendation


def generate_loss_plot(model_history):
    """Generate training/validation loss plot"""
    plt.figure(figsize=(12, 6))

    if not model_history or (len(model_history.get('loss', [])) == 0 and len(model_history.get('val_loss', [])) == 0):
        plt.text(0.5, 0.5, 'No training history available',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes)
    else:
        epochs = range(1, len(model_history['loss']) + 1)
        plt.plot(epochs, model_history['loss'], 'b', label='Training Loss')
        if 'val_loss' in model_history:
            plt.plot(epochs, model_history['val_loss'],
                     'r', label='Validation Loss')

        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f'<img src="data:image/png;base64,{plot_url}">'


if __name__ == "__main__":
    if model is None:
        print("Warning: Model failed to load. Predictions will not work.")
    app.run(debug=True, port=5000)
