#!/bin/bash
set -e

echo "🧪 Running tests..."
pip install pytest pytest-cov
pytest tests/ -v --cov=app
echo "✅ Tests complete!"