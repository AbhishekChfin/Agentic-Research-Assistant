FROM python:3.11-slim

# Install PostgreSQL
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql postgresql-client && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x docker/entrypoint.sh

EXPOSE 8000 5432

ENTRYPOINT ["docker/entrypoint.sh"]
