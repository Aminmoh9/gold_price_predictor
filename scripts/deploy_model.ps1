# deploy_model.ps1
# Run this script after training locally to upload the new model to EC2
# Usage: .\deploy_model.ps1

# ============================================
# CONFIGURATION - Update these values!
# ============================================
$EC2_IP = "YOUR_EC2_PUBLIC_IP"           # e.g., "54.123.45.67"
$KEY_PATH = "path\to\your-key.pem"       # e.g., "C:\Users\Amin\.ssh\gold-predictor.pem"
$EC2_USER = "ubuntu"                      # Default for Ubuntu AMI
$REMOTE_PATH = "/home/ubuntu/gold_price_predictor"

# ============================================
# DO NOT EDIT BELOW THIS LINE
# ============================================

Write-Host "🚀 Gold Price Predictor - Model Deployment Script" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# Check if key file exists
if (!(Test-Path $KEY_PATH)) {
    Write-Host "❌ Error: SSH key not found at $KEY_PATH" -ForegroundColor Red
    exit 1
}

# Check if model files exist
$modelFiles = @(
    "models\gold_lstm_model.keras",
    "models\gold_scaler.pkl",
    "models\model_metadata.json"
)

foreach ($file in $modelFiles) {
    if (!(Test-Path $file)) {
        Write-Host "❌ Error: Model file not found: $file" -ForegroundColor Red
        Write-Host "   Run training script first!" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "`n📤 Uploading model files to EC2..." -ForegroundColor Yellow

# Upload model files
try {
    scp -i $KEY_PATH -o StrictHostKeyChecking=no `
        models/gold_lstm_model.keras `
        models/gold_scaler.pkl `
        models/model_metadata.json `
        "${EC2_USER}@${EC2_IP}:${REMOTE_PATH}/models/"
    
    Write-Host "✅ Model files uploaded successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to upload model files: $_" -ForegroundColor Red
    exit 1
}

# Restart Flask service to load new model
Write-Host "`n🔄 Restarting Flask service..." -ForegroundColor Yellow

try {
    ssh -i $KEY_PATH -o StrictHostKeyChecking=no `
        "${EC2_USER}@${EC2_IP}" `
        "sudo systemctl restart gold-predictor 2>/dev/null || sudo pkill -f 'python.*app.py' && cd ${REMOTE_PATH}/src && nohup python app.py > /dev/null 2>&1 &"
    
    Write-Host "✅ Flask service restarted" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not restart service. You may need to restart manually." -ForegroundColor Yellow
}

# Display model metadata
Write-Host "`n📊 Deployed Model Info:" -ForegroundColor Cyan
Get-Content models\model_metadata.json | ConvertFrom-Json | Format-List

Write-Host "`n🎉 Deployment complete!" -ForegroundColor Green
Write-Host "   API available at: http://${EC2_IP}:5000" -ForegroundColor Cyan
