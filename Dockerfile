FROM python:3.10-slim

WORKDIR /app

# Install system build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Expose Streamlit dashboard port
EXPOSE 8501

# Default launch command: Streamlit live dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
