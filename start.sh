#!/bin/sh
set -e

echo "🚀 SagaMoent start.sh running"

MODEL_DIR="/models"
MODEL_DST="/models/sagacoin_full_model.onnx"

# 🔑 Opret models-mappen hvis den ikke findes
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DST" ]; then
  echo "⬇️ Downloading ONNX model..."
  gdown "$MODEL_URL" -O "$MODEL_DST"
  echo "✅ Model downloaded"
else
  echo "✅ Model already exists"
fi

exec python server_v2.py
