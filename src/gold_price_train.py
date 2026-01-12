# gold_price_train.py

import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import warnings
import json
from datetime import datetime
import mlflow
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# SNS Configuration for Model Degradation Alerts
# ============================================
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")  # Default to us-east-1
MAPE_THRESHOLD = 2.0  # Alert if MAPE exceeds this value

def notify_model_performance(mape, rmse, direction_accuracy, threshold=MAPE_THRESHOLD):
    """Send SNS notification if model performance degrades."""
    if not SNS_TOPIC_ARN:
        print("⚠️ SNS_TOPIC_ARN not set. Skipping notification.")
        return
    
    try:
        if mape > threshold:
            sns = boto3.client("sns", region_name=AWS_REGION)
            message = f"""⚠️ Gold Price Model Performance Alert

MAPE: {mape:.2f}% (Threshold: {threshold}%)
RMSE: ${rmse:.2f}
Direction Accuracy: {direction_accuracy:.1f}%

Action Required: Consider retraining the model or investigating data quality.

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject='⚠️ Gold Model Performance Degraded'
            )
            print(f"📧 Alert sent: MAPE {mape:.2f}% exceeds threshold {threshold}%")
        else:
            print(f"✅ Model performance OK: MAPE {mape:.2f}% within threshold {threshold}%")
    except Exception as e:
        print(f"⚠️ Could not send SNS notification: {e}")

warnings.filterwarnings('ignore')
mlflow.set_experiment("Gold_Price_LSTM")

# Step 1: Download Gold Data
ticker = 'GC=F'
data = yf.download(ticker, period='5y', interval='1d', progress=False)
os.makedirs('../data/raw', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'../data/raw/gold_raw_5y_{timestamp}.csv'
data.to_csv(csv_path)

# Delete old gold_raw_5y_*.csv files except the latest
import glob
csv_files = sorted(glob.glob('../data/raw/gold_raw_5y_*.csv'))
for old_file in csv_files[:-1]:
    try:
        os.remove(old_file)
        print(f"Deleted old CSV: {old_file}")
    except Exception as e:
        print(f"Could not delete {old_file}: {e}")

# Step 2: Prepare Data
prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Balanced hyperparameters: good MAPE (1.30%) + best direction accuracy (52.2%)
LOOKBACK_DAYS = 20  # Balanced config (not 30 which has worse direction accuracy)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK_DAYS)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 3: Build and Train Model (Optimized hyperparameters)
LSTM_UNITS = 100  # Tuned from 50
DROPOUT = 0.2

model = Sequential([
    LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(DROPOUT),
    LSTM(units=LSTM_UNITS, return_sequences=False),
    Dropout(DROPOUT),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

EPOCHS = 60  # Balanced config for better direction accuracy (52.2%)
BATCH_SIZE = 32

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("lookback_days", LOOKBACK_DAYS)
    mlflow.log_param("lstm_units_1", LSTM_UNITS)
    mlflow.log_param("lstm_units_2", LSTM_UNITS)
    mlflow.log_param("dropout", DROPOUT)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    
    # Log metrics
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    mape = mean_absolute_percentage_error(actual, predictions) * 100
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = np.mean(np.abs(actual - predictions))
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    actual_direction = (actual[1:] > actual[:-1]).astype(int)
    pred_direction = (predictions[1:] > predictions[:-1]).astype(int)
    direction_accuracy = (actual_direction == pred_direction).mean() * 100
    mlflow.log_metric("Direction_Accuracy", direction_accuracy)
    
    # Log model with signature to avoid warnings
    from mlflow.models.signature import infer_signature
    import tempfile
    
    # Create signature (skip input_example to avoid file path issues)
    signature = infer_signature(X_train[:1], model.predict(X_train[:1], verbose=0))
    mlflow.keras.log_model(model=model, name="model", signature=signature)
    
    # Log scaler as artifact
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        joblib.dump(scaler, tmp.name)
        mlflow.log_artifact(tmp.name, "scaler")
    # Log metadata
    metadata = {
        'lookback_days': LOOKBACK_DAYS,
        'mape': float(mape),
        'rmse': float(rmse),
        'direction_accuracy': float(direction_accuracy),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tmp:
        json.dump(metadata, tmp, indent=2)
        tmp.flush()
        mlflow.log_artifact(tmp.name, "metadata")

# Step 4: Evaluate Model
predictions_scaled = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mape = mean_absolute_percentage_error(actual, predictions) * 100
rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = np.mean(np.abs(actual - predictions))
actual_direction = (actual[1:] > actual[:-1]).astype(int)
pred_direction = (predictions[1:] > predictions[:-1]).astype(int)
direction_accuracy = (actual_direction == pred_direction).mean() * 100

print(f"\n🎯 MODEL PERFORMANCE METRICS:")
print("=" * 50)
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"Target: < 2.00%")
print(f"Status: {'✅ PASS' if mape < 2.0 else '⚠️ NEEDS IMPROVEMENT'}")
print("-" * 50)
print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
print(f"MAE (Mean Absolute Error): ${mae:.2f}")
print(f"Direction Accuracy: {direction_accuracy:.1f}%")
print("=" * 50)

# Step 5: Save Model, Scaler, and Metadata
os.makedirs('models', exist_ok=True)
model.save('models/gold_lstm_model.keras')  # Use native Keras format
joblib.dump(scaler, 'models/gold_scaler.pkl')
metadata = {
    'lookback_days': LOOKBACK_DAYS,
    'mape': float(mape),
    'rmse': float(rmse),
    'direction_accuracy': float(direction_accuracy),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Model, scaler, and metadata saved.")

# Step 6: Check for Model Degradation and Send Alert
notify_model_performance(mape, rmse, direction_accuracy)