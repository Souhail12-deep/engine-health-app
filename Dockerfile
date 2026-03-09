FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY app/ ./app/
COPY routes/ ./routes/
COPY services/ ./services/
COPY utils/ ./utils/
COPY config.py ./
COPY templates/ ./templates/
COPY static/ ./static/

# Copy models and data
COPY models/ ./models/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app

# Debug: List contents
RUN echo "=== Container contents ===" && \
    ls -la && \
    echo "=== Models directory ===" && \
    ls -la models/ && \
    echo "=== Models/anodet_models ===" && \
    ls -la models/anodet_models/ && \
    echo "=== Models/rul_models ===" && \
    ls -la models/rul_models/

EXPOSE 5000

CMD ["python", "app/app.py"]