#!/usr/bin/env bash
# Production Validation - Direct Execution
# File: validate-production.sh

set -e

echo "🔧 PRODUCTION VALIDATION"
echo "======================="

# Force GPU environment
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_GPU_LAYERS=35

# Create logs directory
mkdir -p logs

# Test 1: Ollama service
echo "Testing Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama: RUNNING"
    OLLAMA_STATUS="RUNNING"
else
    echo "❌ Ollama: OFFLINE"
    OLLAMA_STATUS="OFFLINE"
fi

# Test 2: Import Gemma3 Legal
echo "Testing Gemma3 Legal model..."
GEMMA_RESPONSE=$(curl -s -w "%{http_code}" -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3-legal","prompt":"Test","stream":false}' 2>/dev/null | tail -c 3)

if [ "$GEMMA_RESPONSE" = "200" ]; then
    echo "✅ Gemma3 Legal: OPERATIONAL"
    GEMMA_STATUS="OPERATIONAL"
else
    echo "❌ Gemma3 Legal: OFFLINE"
    GEMMA_STATUS="OFFLINE"
fi

# Test 3: SvelteKit dev server
echo "Testing SvelteKit dev server..."
if curl -s -I http://localhost:5173 | grep -q "200\|404"; then
    echo "✅ SvelteKit: RUNNING" 
    SVELTE_STATUS="RUNNING"
else
    echo "❌ SvelteKit: OFFLINE"
    SVELTE_STATUS="OFFLINE"
fi

# Test 4: Evidence Synthesis API
echo "Testing Synthesis API..."
SYNTH_RESPONSE=$(curl -s -w "%{http_code}" -X POST http://localhost:5173/api/evidence/synthesize \
  -H "Content-Type: application/json" \
  -d '{"evidenceIds":["test"],"synthesisType":"correlation","caseId":"test","title":"Test"}' 2>/dev/null | tail -c 3)

if [ "$SYNTH_RESPONSE" = "401" ] || [ "$SYNTH_RESPONSE" = "200" ]; then
    echo "✅ Synthesis API: ACCESSIBLE"
    SYNTH_STATUS="ACCESSIBLE"
else
    echo "❌ Synthesis API: OFFLINE"
    SYNTH_STATUS="OFFLINE"
fi

# Generate production report
cat > logs/production-validation.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "validation_results": {
    "ollama": "$OLLAMA_STATUS",
    "gemma3_legal": "$GEMMA_STATUS",
    "sveltekit": "$SVELTE_STATUS", 
    "synthesis_api": "$SYNTH_STATUS"
  },
  "endpoints": {
    "ai_assistant": "http://localhost:5173/ai-assistant",
    "synthesis_api": "http://localhost:5173/api/evidence/synthesize",
    "rag_studio": "http://localhost:5173/api/enhanced-rag/query"
  },
  "next_actions": [
    "Start Ollama: ollama serve",
    "Import model: ollama create gemma3-legal -f local-models/Modelfile.gemma3-legal", 
    "Start dev server: npm run dev",
    "Force GPU: export CUDA_VISIBLE_DEVICES=0"
  ]
}
EOF

echo ""
echo "📊 VALIDATION COMPLETE"
echo "Ollama: $OLLAMA_STATUS"
echo "Gemma3: $GEMMA_STATUS"
echo "SvelteKit: $SVELTE_STATUS"
echo "Synthesis: $SYNTH_STATUS"
echo ""
echo "Report: logs/production-validation.json"

# Auto-start if needed
if [ "$OLLAMA_STATUS" = "OFFLINE" ]; then
    echo "🚀 Auto-starting Ollama with GPU..."
    nohup ollama serve > logs/ollama.log 2>&1 &
    sleep 3
fi

if [ "$GEMMA_STATUS" = "OFFLINE" ] && [ "$OLLAMA_STATUS" = "RUNNING" ]; then
    echo "📥 Auto-importing Gemma3 Legal..."
    cd local-models && ollama create gemma3-legal -f Modelfile.gemma3-legal
fi
