#!/bin/sh
set -e

echo "🚀 SagaMoent entrypoint"

if [ ! -f /models/sagacoin_full_model.onnx ]; then
  echo "📦 Model missing in volume – copying..."
  cp /app/model_src/sagacoin_full_model.onnx /models/
else
  echo "✔ Model already exists in volume"
fi

exec python server_v2.py
