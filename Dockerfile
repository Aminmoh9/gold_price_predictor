# Gold Price Predictor - Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow and healthcheck
RUN apt-get update && apt-get install -y \
    gcc \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (boto3 already included in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy only LSTM model files (used by Flask API)
COPY models/gold_lstm_model.keras ./models/
COPY models/gold_scaler.pkl ./models/
COPY models/model_metadata.json ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the Flask application
CMD ["python", "src/app.py"]
