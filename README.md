# Gold Price Predictor Project

## Overview
This project predicts Gold/USD prices using machine learning and MLOps best practices, with cloud deployment on AWS. It includes data collection, model training, API deployment, automation, monitoring, and notification features.

### Model Performance
| Model | MAPE | RMSE | Direction Accuracy |
|-------|------|------|-------------------|
| **Ridge (sklearn)** | 1.01% 🏆 | $49.66 | 53.3% |
| LSTM (Deep Learning) | 1.49% | $72.05 | 51.0% |

✅ Both models pass the target of **MAPE < 2%**

---

## Project Structure
```
├── data/
│   └── raw/                    # Raw gold price data (timestamped CSV, only latest kept)
├── guides/                     # All documentation and guides
│   ├── cloud_automation_guide.md
│   ├── docker_guide.md
│   ├── ec2_deployment_guide.md
│   ├── ec2_setup_guide.md
│   ├── feature_engineering_guide.md  # Feature & model explanation
│   ├── implementation_plan.md
│   ├── lambda_automation_guide.md
│   ├── presentation_outline.md
│   ├── project_structure_guide.md
│   ├── s3_model_storage_guide.md
│   └── serverless_automation_guide.md
├── mlruns/                     # MLflow experiment tracking (unified)
├── models/                     # Saved models and metadata
│   ├── gold_lstm_model.keras   # LSTM model (used by Flask API)
│   ├── gold_scaler.pkl         # LSTM scaler
│   ├── model_metadata.json     # LSTM metrics
│   ├── gold_sklearn_model.pkl  # Best sklearn model (Ridge)
│   ├── gold_sklearn_scaler.pkl # sklearn scaler
│   ├── sklearn_metadata.json   # sklearn metrics
│   └── best_hyperparameters.json # Tuned hyperparameters
├── notebooks/                  # Jupyter notebooks
│   ├── gold_price_prediction.ipynb      # Main LSTM training
│   ├── comprehensive_training.ipynb     # sklearn models training
│   ├── all_models_comparison.ipynb      # 5-model comparison
│   ├── hyperparameter_tuning.ipynb      # Hyperparameter optimization
│   └── exploration/                     # Experimental notebooks
├── scripts/                    # Deployment scripts (PowerShell, Bash)
├── src/                        # Source code
│   ├── app.py                  # Flask API server
│   ├── gold_price_train.py     # LSTM model training script
│   └── templates/              # Web UI templates (index.html)
├── .github/workflows/          # GitHub Actions CI/CD
├── Dockerfile                  # Docker containerization
├── docker-compose.yml          # Docker Compose config
├── .dockerignore               # Docker ignore rules
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project overview
```

---

## ML Pipeline
1. **Data Collection:** Fetch gold price data using yfinance, save to `data/raw/` (timestamped, only latest kept)
2. **Preprocessing:** Clean, scale, and prepare data for modeling
3. **Model Training:** Train LSTM model (locally or on cloud), save to `models/` as `.keras`
4. **Experiment Tracking:** Log metrics and parameters with MLflow (`mlruns/`)
5. **Model Deployment:** Serve predictions via Flask API (`src/app.py`) on EC2 or Docker
6. **Web Interface:** User-friendly UI at `/` endpoint
7. **Automation & Alerts (Serverless - No EC2 24/7 required):**
   - **Price Alerts:** AWS Lambda + EventBridge (daily) → checks price changes → SNS email
   - **Scheduled Retraining:** GitHub Actions (weekly Sunday 2 AM) → trains model → uploads to S3
   - **Model Degradation:** Training script → SNS alert if MAPE > 2%
8. **CI/CD:** GitHub Actions for automated retraining and deployment (`.github/workflows/`)

---

## Getting Started
1. Clone the repo and set up your Python environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/gold_price_train.py`
4. Deploy API: `python src/app.py` or use Docker (see `guides/docker_guide.md`)
5. Access web UI at `http://localhost:5000/`
6. Set up automation and notifications (see `guides/lambda_automation_guide.md`)
7. For cloud automation, follow `guides/cloud_automation_guide.md`

---

## Documentation
All guides are in the `guides/` folder:
- **feature_engineering_guide.md:** 📚 Feature engineering, Ridge model, direction accuracy explained
- **serverless_automation_guide.md:** Complete serverless automation setup (Lambda, GitHub Actions, SNS)
- **docker_guide.md:** Docker deployment steps
- **ec2_deployment_guide.md:** EC2 deployment instructions
- **ec2_setup_guide.md:** EC2 initial setup
- **s3_model_storage_guide.md:** S3 model storage and retrieval
- **lambda_automation_guide.md:** Lambda automation and notifications
- **cloud_automation_guide.md:** CI/CD retraining and redeployment
- **implementation_plan.md:** Full project plan and checklist
- **presentation_outline.md:** Presentation structure
- **project_structure_guide.md:** Codebase overview

## Notebooks
- **gold_price_prediction.ipynb:** Main LSTM model training with MLflow tracking
- **comprehensive_training.ipynb:** sklearn models (Ridge, Random Forest, XGBoost, Gradient Boosting)
- **all_models_comparison.ipynb:** Compare all 5 models side-by-side with metrics
- **hyperparameter_tuning.ipynb:** Systematic hyperparameter optimization

---

## License
MIT

---

**For questions or improvements, open an issue or pull request!**
