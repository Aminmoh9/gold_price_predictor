#!/bin/bash
# train_and_deploy.sh
# Complete workflow: Train model locally, then deploy to EC2
# Usage: ./scripts/train_and_deploy.sh

echo "🏋️ Gold Price Predictor - Train & Deploy"
echo "========================================="

# Step 1: Train the model
echo ""
echo "📚 Step 1: Training model..."
cd src
python gold_price_train.py

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ Training complete!"

# Step 2: Deploy to EC2
cd ..
echo ""
echo "📤 Step 2: Deploying to EC2..."
./scripts/deploy_model.sh

echo ""
echo "🎉 Train & Deploy complete!"
