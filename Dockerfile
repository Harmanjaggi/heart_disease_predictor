FROM python:3.9-slim

WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# --- NEW: Run tests during build ---
# If these fail, the build stops here.
RUN pytest tests/

CMD ["python", "pipeline_runner.py"]
