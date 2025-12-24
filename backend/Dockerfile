# Sharp Sports Predictor - Railway Backend
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory (for Railway volume mount)
RUN mkdir -p /data

# Verify files are present (build-time check)
RUN echo "=== Verifying files ===" && \
    ls -la /app/*.py && \
    ls -la /app/*.pkl && \
    ls -la /app/*.csv && \
    echo "=== All files present ==="

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Note: Railway uses its own health check from railway.json, not Docker HEALTHCHECK

# Run with Python directly for better error messages
CMD ["sh", "-c", "cd /app && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
