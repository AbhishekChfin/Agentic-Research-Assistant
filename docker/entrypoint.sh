#!/bin/bash
set -e

# Start PostgreSQL
service postgresql start

# Create database and user (idempotent)
su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER:-app}'\" | grep -q 1 || psql -c \"CREATE USER ${POSTGRES_USER:-app} WITH PASSWORD '${POSTGRES_PASSWORD:-local}';\"" 2>/dev/null
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='${POSTGRES_DB:-appdb}'\" | grep -q 1 || psql -c \"CREATE DATABASE ${POSTGRES_DB:-appdb} OWNER ${POSTGRES_USER:-app};\"" 2>/dev/null

# Export DATABASE_URL for the app
export DATABASE_URL="postgresql://${POSTGRES_USER:-app}:${POSTGRES_PASSWORD:-local}@127.0.0.1:5432/${POSTGRES_DB:-appdb}"

echo "Database ready at $DATABASE_URL"

# Launch FastAPI
exec uvicorn main:app --host 0.0.0.0 --port 8000
