FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy files
COPY serving/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ .
COPY mlruns/ ./mlruns/

# Expose port
EXPOSE 4723

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4723"]
