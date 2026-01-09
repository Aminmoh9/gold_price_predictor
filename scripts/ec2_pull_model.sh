#!/bin/bash
# ec2_pull_model.sh
# Run this on EC2 to pull the latest model from S3
# This is for Option B (GitHub Actions + S3)
# Add to crontab or run on startup

# ============================================
# CONFIGURATION
# ============================================
S3_BUCKET="your-gold-predictor-bucket"
APP_DIR="/home/ubuntu/gold_price_predictor"
MODEL_DIR="${APP_DIR}/models"
LOG_FILE="/var/log/gold-predictor-sync.log"

# ============================================
# SCRIPT
# ============================================

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

log "🚀 Starting model sync from S3..."

# Create model directory if not exists
mkdir -p ${MODEL_DIR}

# Download latest model from S3
log "📥 Downloading latest model..."
aws s3 sync s3://${S3_BUCKET}/models/latest/ ${MODEL_DIR}/ --delete

# Verify model files exist
if [ -f "${MODEL_DIR}/gold_lstm_model.keras" ]; then
    log "✅ Model downloaded successfully"
    log "📊 Model metadata:"
    cat ${MODEL_DIR}/model_metadata.json | tee -a $LOG_FILE
    
    # Restart Flask to load new model
    log "🔄 Restarting Flask service..."
    sudo systemctl restart gold-predictor 2>/dev/null || \
        (pkill -f 'python.*app.py'; cd ${APP_DIR}/src && nohup python app.py > /dev/null 2>&1 &)
    
    log "🎉 Model sync complete!"
else
    log "❌ Model download failed!"
    exit 1
fi
