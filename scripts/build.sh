#!/bin/bash
set -e

echo "🔨 Building Docker image..."
docker build -t engine-health-app:latest .
echo "✅ Build complete!"