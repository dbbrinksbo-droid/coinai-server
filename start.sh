#!/bin/sh
set -e

echo "🚨🚨🚨 THIS IS THE NEW start.sh 🚨🚨🚨"
echo "PWD=$(pwd)"
echo "LS / ="
ls -la /

MODEL_DST="/tmp/sagacoin_full_model.onnx"

if [ ! -f "$MODEL_DST" ]; then
  echo "⬇️ Downloading ONNX model to /tmp..."
  gdown "$MODEL_URL" -O "$MODEL_DST"
  echo "✅ Model downloaded to /tmp"
else
  echo "✅ Model already exists in /tmp"
fi

exec python server_v2.py
