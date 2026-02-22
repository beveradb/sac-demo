#!/bin/bash
set -e

echo "=== SaC Demo Companion Server ==="
echo "sac2c version: $(sac2c -V 2>&1 | head -1)"
echo "Listening on port 7227..."

exec uvicorn server:app --host 0.0.0.0 --port 7227 --app-dir /app
