FROM python:3.11-slim

WORKDIR /app

# Install build dependencies and JDK for PySpark
RUN apt-get update && apt-get install -y \
    gcc \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and raw data
COPY src/ ./src/
COPY data/raw/ ./data/raw/
COPY train.py .

# Create directory structures
RUN mkdir -p models data/processed data/campaigns

# Train all models and generate the reports during build time
RUN python train.py

EXPOSE 8000

# Set python path environment variables
ENV PYTHONPATH="/app:/app/src"

CMD ["uvicorn", "src.cvm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
