#!/usr/bin/env bash
# Direct System Validation - No Dependencies
# File: validate-system.sh

echo "🔧 DIRECT SYSTEM VALIDATION"
echo "=========================="

# Test SvelteKit dev server
echo "Testing SvelteKit dev server..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5173 | grep -q "200\|404"; then
    echo "✅ SvelteKit dev server: RUNNING"
    SVELTEKIT_STATUS="RUNNING"
else
    echo "❌ SvelteKit dev server: OFFLINE"
    SVELTEKIT_STATUS="OFFLINE"
fi

# Test Ollama service
echo "Testing Ollama service..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q "200"; then
    echo "✅ Ollama service: RUNNING"
    OLLAMA_STATUS="RUNNING"
else
    echo "❌ Ollama service: OFFLINE"
    OLLAMA_STATUS="OFFLINE"
fi

# Test Evidence Synthesis API
echo "Testing Synthesis API..."
SYNTHESIS_RESPONSE=$(curl -s -w "%{http_code}" -X POST http://localhost:5173/api/evidence/synthesize \
  -H "Content-Type: application/json" \
  -d '{"evidenceIds":["test"],"synthesisType":"correlation","caseId":"test","title":"Test"}' 2>/dev/null)

if echo "$SYNTHESIS_RESPONSE" | grep -q "401\|404\|200"; then
    echo "✅ Synthesis API: ACCESSIBLE"
    SYNTHESIS_STATUS="ACCESSIBLE"
else
    echo "❌ Synthesis API: OFFLINE"
    SYNTHESIS_STATUS="OFFLINE"
fi

# Generate validation report
cat > logs/validation-report.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "services": {
    "sveltekit": "$SVELTEKIT_STATUS",
    "ollama": "$OLLAMA_STATUS", 
    "synthesis": "$SYNTHESIS_STATUS"
  },
  "recommendations": [
    "Start Ollama: ollama serve",
    "Start SvelteKit: npm run dev",
    "Import model: ollama create gemma3-legal -f local-models/Modelfile.gemma3-legal"
  ]
}
EOF

echo ""
echo "📊 VALIDATION COMPLETE"
echo "SvelteKit: $SVELTEKIT_STATUS"
echo "Ollama: $OLLAMA_STATUS"
echo "Synthesis: $SYNTHESIS_STATUS"
echo ""
echo "Report: logs/validation-report.json"
