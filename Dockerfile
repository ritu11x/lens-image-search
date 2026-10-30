FROM python:3.11-slim

# Create a non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install minimal OS dependencies for Pillow and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data files
COPY --chown=user:user . .

# Set environment variables for Hugging Face Spaces
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface

# Ensure directories are writable by user 1000
RUN mkdir -p /home/user/.cache/huggingface /app/images /app/thumbnails && \
    chown -R user:user /home/user /app

USER user

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
