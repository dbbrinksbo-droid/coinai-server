#!/bin/sh
set -e

echo "🚀 SagaMoent start.sh running"

MODEL_SRC="/app/sagacoin_full_model.onnx"
MODEL_DST="/models/sagacoin_full_model.onnx"

if [ ! -f "$MODEL_DST" ]; then
  echo "📥 Model not found in volume — bootstrapping"
  if [ -f "$MODEL_SRC" ]; then
    cp "$MODEL_SRC" "$MODEL_DST"
    echo "✅ Model copied to volume"
  else
    echo "❌ Model missing in image AND volume"
  fi
else
  echo "✅ Model already exists in volume"
fi

exec python server_v2.py
