import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import json
import yfinance as yf
from datetime import datetime, timedelta
import os

# Constants
SEQUENCE_LENGTH = 100
DATA_PATH = 'BBRI.JK.NEW.csv'
STEP = 5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8


def create_model():
    """Create and save LSTM model following notebook structure"""
    try:
        # Load data
        print("Loading and preparing data...")
        if not os.path.exists(DATA_PATH):
            # Download data if file doesn't exist
            print("Downloading historical stock data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)  # 5 years of data

            ticker = yf.Ticker("BBRI.JK")
            stock_data = ticker.history(start=start_date, end=end_date)
            stock_data = stock_data.reset_index()
            stock_data.to_csv(DATA_PATH, index=False)
        else:
            stock_data = pd.read_csv(DATA_PATH)

        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)

        # Remove duplicates
        stock_data = stock_data.drop_duplicates()

        # Handle missing values with interpolation
        stock_data = stock_data.asfreq('B')
        stock_data.interpolate(method='linear', inplace=True)

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data[['Close']].dropna())

        # Create sequences
        print("Creating sequences...")
        x_data = []
        y_data = []

        for i in range(SEQUENCE_LENGTH, len(scaled_data), STEP):
            sequence = scaled_data[i-SEQUENCE_LENGTH:i]
            target = scaled_data[i]
            x_data.append(sequence)
            y_data.append(target)

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Split data
        print("Splitting data...")
        train_size = int(len(x_data) * TRAIN_SPLIT)
        x_train, y_train = x_data[:train_size], y_data[:train_size]
        x_test, y_test = x_data[train_size:], y_data[train_size:]

        # Create validation set
        val_size = int(len(x_train) * 0.2)
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]

        print(f"Data shapes:")
        print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
        print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

        # Create model
        print("Creating LSTM model...")
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
            Dropout(0.1),
            LSTM(64),  # Second LSTM layer
            Dropout(0.1),
            Dense(1)  # Output layer
        ])

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                      loss='mean_squared_error')

        # Train model with early stopping
        print("Training model...")
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1
        )

        # Save model
        print("Saving model...")
        model.save('model.keras')

        # Save training history
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }

        with open('model_history.json', 'w') as f:
            json.dump(history_dict, f)

        # Calculate and display metrics
        predictions = model.predict(x_test)
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test)

        mape = np.mean(
            np.abs((inv_predictions - inv_y_test) / inv_y_test)) * 100
        accuracy = 100 - mape

        print("\nModel Performance:")
        print(f"MAPE: {mape:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")

        print("\nModel and history saved successfully!")
        return True

    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return False


if __name__ == "__main__":
    create_model()
