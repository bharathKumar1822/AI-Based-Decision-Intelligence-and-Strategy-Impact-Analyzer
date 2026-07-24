# Use Python 3.11 slim for smaller image size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p chroma_db data

# Expose port
EXPOSE 5000

# Environment defaults
ENV PORT=5000 \
    OLLAMA_URL=http://ollama:11434 \
    OLLAMA_MODEL=llama3 \
    CHROMA_PERSIST_DIR=/app/chroma_db

# Start with gunicorn (production) or flask dev server
CMD ["python", "run.py"]
