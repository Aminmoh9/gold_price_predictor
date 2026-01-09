#!/bin/bash
# EC2 Startup Script - Downloads latest model from S3 on boot
# Add this to EC2 User Data or run on startup via systemd

set -e

# Configuration
S3_BUCKET="your-gold-predictor-bucket"  # Replace with your bucket name
APP_DIR="/home/ubuntu/gold_price_predictor"
MODEL_DIR="${APP_DIR}/models"

echo "🚀 Starting EC2 model sync..."

# Create model directory if not exists
mkdir -p ${MODEL_DIR}

# Download latest model from S3
echo "📥 Downloading latest model from S3..."
aws s3 sync s3://${S3_BUCKET}/models/latest/ ${MODEL_DIR}/ --delete

# Verify model files exist
if [ -f "${MODEL_DIR}/gold_lstm_model.keras" ]; then
    echo "✅ Model downloaded successfully"
    cat ${MODEL_DIR}/model_metadata.json
else
    echo "❌ Model download failed!"
    exit 1
fi

# Start the Flask API
echo "🌐 Starting Flask API..."
cd ${APP_DIR}/src
source ../venv/bin/activate  # Adjust path to your virtualenv
python app.py &

echo "🎉 EC2 startup complete!"
