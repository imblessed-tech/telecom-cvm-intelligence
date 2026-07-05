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

# Copy application source code and artifacts
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/campaigns/ ./data/campaigns/

EXPOSE 8000

# Set python path environment variables
ENV PYTHONPATH="/app:/app/src"

CMD ["uvicorn", "src.cvm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
