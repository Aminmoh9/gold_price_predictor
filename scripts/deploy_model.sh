#!/bin/bash
# deploy_model.sh
# Run this script after training locally to upload the new model to EC2
# Usage: ./scripts/deploy_model.sh

# ============================================
# CONFIGURATION - Update these values!
# ============================================
EC2_IP="YOUR_EC2_PUBLIC_IP"              # e.g., "54.123.45.67"
KEY_PATH="path/to/your-key.pem"          # e.g., "~/.ssh/gold-predictor.pem"
EC2_USER="ubuntu"                         # Default for Ubuntu AMI
REMOTE_PATH="/home/ubuntu/gold_price_predictor"

# ============================================
# DO NOT EDIT BELOW THIS LINE
# ============================================

echo "🚀 Gold Price Predictor - Model Deployment Script"
echo "================================================="

# Check if key file exists
if [ ! -f "$KEY_PATH" ]; then
    echo "❌ Error: SSH key not found at $KEY_PATH"
    exit 1
fi

# Check if model files exist
for file in models/gold_lstm_model.keras models/gold_scaler.pkl models/model_metadata.json; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Model file not found: $file"
        echo "   Run training script first!"
        exit 1
    fi
done

echo ""
echo "📤 Uploading model files to EC2..."

# Upload model files
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
    models/gold_lstm_model.keras \
    models/gold_scaler.pkl \
    models/model_metadata.json \
    "${EC2_USER}@${EC2_IP}:${REMOTE_PATH}/models/"

if [ $? -eq 0 ]; then
    echo "✅ Model files uploaded successfully"
else
    echo "❌ Failed to upload model files"
    exit 1
fi

# Restart Flask service
echo ""
echo "🔄 Restarting Flask service..."

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no \
    "${EC2_USER}@${EC2_IP}" \
    "sudo systemctl restart gold-predictor 2>/dev/null || (sudo pkill -f 'python.*app.py'; cd ${REMOTE_PATH}/src && nohup python app.py > /dev/null 2>&1 &)"

echo "✅ Flask service restarted"

# Display model metadata
echo ""
echo "📊 Deployed Model Info:"
cat models/model_metadata.json

echo ""
echo "🎉 Deployment complete!"
echo "   API available at: http://${EC2_IP}:5000"
