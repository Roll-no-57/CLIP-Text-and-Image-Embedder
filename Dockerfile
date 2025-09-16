# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_v2.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_v2.txt

# Copy application code
COPY app.py .

# Create cache directories and set permissions
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers && \
    useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

# Expose port 8080 (Google Cloud Run requirement)
EXPOSE 8080

# Health check with longer timeout for model loading
HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application with increased timeout for Cloud Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "30"]
