"""
Flask API for Gold Price Prediction
"""

# Suppress TensorFlow warnings before importing
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, render_template, request as flask_request
from flask_cors import CORS
import numpy as np
import yfinance as yf
import joblib
from tensorflow import keras
import json
from datetime import datetime

app = Flask(__name__, template_folder=os.path.abspath('src/templates'))

# ============================================================================
# S3 MODEL DOWNLOAD (for cloud deployment)
# ============================================================================
S3_BUCKET = os.environ.get('S3_MODEL_BUCKET', None)  # Set in EC2 environment
S3_MODEL_PREFIX = 'models/'
LOCAL_MODEL_DIR = 'models/'

def download_models_from_s3():
    """Download model files from S3 on startup (only if S3_MODEL_BUCKET is set)"""
    if not S3_BUCKET:
        logger.info("S3_MODEL_BUCKET not set, using local models")
        return False
    
    try:
        import boto3
        s3 = boto3.client('s3')
        
        # Create local models directory if it doesn't exist
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        files_to_download = [
            'gold_lstm_model.keras',
            'gold_scaler.pkl',
            'model_metadata.json'
        ]
        
        for filename in files_to_download:
            s3_key = f"{S3_MODEL_PREFIX}{filename}"
            local_path = os.path.join(LOCAL_MODEL_DIR, filename)
            
            logger.info(f"Downloading {filename} from S3...")
            s3.download_file(S3_BUCKET, s3_key, local_path)
            logger.info(f"✓ Downloaded {filename}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return False

# Download models from S3 if configured
download_models_from_s3()

# ============================================================================
# LOAD MODEL AND ARTIFACTS AT STARTUP
# ============================================================================
logger.info("Starting Gold Price Prediction API")

# Load saved model and scaler
try:
    # Support both new .keras format and legacy .h5 format
    MODEL_PATH_KERAS = 'models/gold_lstm_model.keras'
    MODEL_PATH_H5 = 'models/gold_lstm_model.h5'
    SCALER_PATH = 'models/gold_scaler.pkl'
    METADATA_PATH = 'models/model_metadata.json'
    
    # Try loading .keras first, fall back to .h5
    if os.path.exists(MODEL_PATH_KERAS):
        MODEL_PATH = MODEL_PATH_KERAS
    else:
        MODEL_PATH = MODEL_PATH_H5
    
    model = keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded from: {MODEL_PATH}")
    
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded successfully")
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    LOOKBACK_DAYS = metadata['lookback_days']
    logger.info(f"Metadata loaded (lookback days: {LOOKBACK_DAYS})")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

logger.info("API Ready!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_recent_gold_data(days=100):
    """
    Fetch recent gold futures price data from Yahoo Finance (GC=F).

    Args:
        days (int): Number of recent days to fetch (should be >= lookback_days).

    Returns:
        numpy.ndarray: Array of closing prices for the last `days` days.
    """
    try:
        ticker = 'GC=F'
        data = yf.download(ticker, period=f'{days}d', interval='1d', progress=False, auto_adjust=False)
        
        if len(data) == 0:
            raise ValueError("No data downloaded")
        
        prices = data['Close'].dropna().values
        return prices, data

    except Exception as e:
        raise ValueError(f"Failed to fetch gold futures data: {str(e)}")


def predict_next_day(recent_prices):
    """
    Predict next day's gold price using LSTM model
    
    Args:
        recent_prices: numpy array of recent closing prices (at least LOOKBACK_DAYS)
    
    Returns:
        dict with prediction results
    """
    if len(recent_prices) < LOOKBACK_DAYS:
        raise ValueError(f"Need at least {LOOKBACK_DAYS} days of data, got {len(recent_prices)}")
    
    # Get last LOOKBACK_DAYS prices
    recent_data = recent_prices[-LOOKBACK_DAYS:]
    
    # Scale the data (same way as training)
    scaled_data = scaler.transform(recent_data.reshape(-1, 1))
    
    # Reshape for LSTM: (1, lookback_days, 1)
    X_input = scaled_data.reshape(1, LOOKBACK_DAYS, 1)
    
    # Make prediction
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # Inverse transform to get actual price
    predicted_price = scaler.inverse_transform(prediction_scaled)[0][0]
    
    # Calculate metrics
    current_price = float(recent_prices[-1].item())
    predicted_price = float(predicted_price.item())   
    change = float(predicted_price - current_price)  
    change_percent = float((change / current_price) * 100)
    direction = 'UP ↗' if change > 0 else 'DOWN ↘' if change < 0 else 'FLAT →'
    
    # Prepare result
    result = {
        'success': True,
        'current_price': float(current_price),
        'predicted_price': float(predicted_price),
        'change': float(change),
        'change_percent': float(change_percent),
        'direction': direction,
        'model_info': {
            'type': 'LSTM',
            'mape': metadata.get('mape', 'N/A'),
            'lookback_days': LOOKBACK_DAYS
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return result


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home - Web UI"""
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'Gold Price Prediction API',
        'version': '1.0',
        'model': 'LSTM',
        'endpoints': {
            '/api': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Get next day price prediction',
            '/predict/auto': 'GET - Auto-fetch data and predict'
        },
        'status': 'running'
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict next day gold price
    
    Expected JSON body:
    {
        "prices": [2045.5, 2048.2, 2050.1, ...]  // Array of recent prices
    }
    
    OR just send empty {} and it will auto-fetch data
    """
    try:
        data = request.get_json()
        
        if data and 'prices' in data:
            prices = np.array(data['prices'])
        else:
            prices, _ = fetch_recent_gold_data(days=100)
        
        result = predict_next_day(prices)
        logger.info(f"Prediction: ${result['predicted_price']:.2f} ({result['direction']})")
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }), 500


@app.route('/predict/auto', methods=['GET'])
def predict_auto():
    """
    Simplified endpoint - automatically fetches data and predicts
    No input needed, just call this endpoint!
    """
    try:
        prices, _ = fetch_recent_gold_data(days=100)
        result = predict_next_day(prices)
        logger.info(f"Prediction: ${result['predicted_price']:.2f} ({result['direction']})")
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict_auto', methods=['POST'])
def web_predict_auto():
    try:
        prices, raw_data = fetch_recent_gold_data(days=100)
        result = predict_next_day(prices)
        
        # Prepare price history for display (all available data)
        price_history = []
        for date, row in raw_data.iterrows():
            close_price = float(row['Close'].item()) if hasattr(row['Close'], 'item') else float(row['Close'])
            price_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(close_price, 2)
            })
        
        # Calculate daily changes
        for i in range(1, len(price_history)):
            change = price_history[i]['price'] - price_history[i-1]['price']
            price_history[i]['change'] = round(change, 2)
        if price_history:
            price_history[0]['change'] = 0
        
        return render_template('index.html', result=result, price_history=price_history)
    except Exception as e:
        return render_template('index.html', error=str(e))


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Flask Server on http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
