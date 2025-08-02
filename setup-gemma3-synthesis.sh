#!/bin/bash
# Gemma3 Legal Model Import & Synthesis Testing
# Run: ./setup-gemma3-synthesis.sh

set -e

echo "🔄 Importing Gemma3 Legal Model..."

# Import GGUF model into Ollama
cd C:/Users/james/Desktop/deeds-web/deeds-web-app/local-models
ollama create gemma3-legal -f Modelfile.gemma3-legal

# Verify model is loaded
ollama list | grep gemma3-legal || echo "Model import failed"

# Pull nomic embeddings for RAG
ollama pull nomic-embed-text

echo "✅ Models ready. Testing synthesis pipeline..."

# Test synthesis API
cd ../
node test-gemma3-synthesis.mjs

echo "🔗 Starting enhanced RAG integration..."
