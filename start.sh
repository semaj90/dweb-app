#!/bin/bash

echo "🔍 Checking for GPU access..."

if nvidia-smi &>/dev/null; then
  echo "✅ GPU detected. Starting Ollama with GPU support..."
  exec ollama serve
else
  echo "⚠️ No GPU found. Falling back to CPU mode..."
  export OLLAMA_NO_GPU=1

  # Try to start Ollama in CPU mode
  if ollama serve; then
    echo "✅ CPU fallback succeeded."
    exit 0
  else
    echo "❌ Error: Neither GPU nor CPU Ollama could start."
    exit 1
  fi
fi
