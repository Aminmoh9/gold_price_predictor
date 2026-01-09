# train_and_deploy.ps1
# Complete workflow: Train model locally, then deploy to EC2
# Usage: .\train_and_deploy.ps1

Write-Host "🏋️ Gold Price Predictor - Train & Deploy" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Step 1: Train the model
Write-Host "`n📚 Step 1: Training model..." -ForegroundColor Yellow
Set-Location src
python gold_price_train.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Training complete!" -ForegroundColor Green

# Step 2: Deploy to EC2
Set-Location ..
Write-Host "`n📤 Step 2: Deploying to EC2..." -ForegroundColor Yellow
& .\scripts\deploy_model.ps1

Write-Host "`n🎉 Train & Deploy complete!" -ForegroundColor Green
