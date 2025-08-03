#!/bin/bash

# Run all Playwright tests for the RAG application stack

echo "🚀 Starting comprehensive test suite for RAG application..."
echo "=================================================="

# Check if Ollama is running
echo "📍 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running. Starting Ollama..."
    npm run ollama:start
    sleep 5
fi

# Check if the dev server is running
echo "📍 Checking dev server..."
if curl -s http://localhost:5175 > /dev/null; then
    echo "✅ Dev server is running"
else
    echo "⚠️  Dev server not running. It will be started by Playwright."
fi

# Run tests in sequence with detailed reporting
echo ""
echo "🧪 Running test suites..."
echo "=================================================="

# 1. Ollama Integration Tests
echo ""
echo "1️⃣ Running Ollama Integration Tests..."
npx playwright test ollama-integration.spec.ts --reporter=list

# 2. RAG System Tests
echo ""
echo "2️⃣ Running RAG System Tests..."
npx playwright test rag-system.spec.ts --reporter=list

# 3. PostgreSQL and pgvector Tests
echo ""
echo "3️⃣ Running PostgreSQL and pgvector Tests..."
npx playwright test postgresql-pgvector.spec.ts --reporter=list

# 4. Drizzle ORM Tests
echo ""
echo "4️⃣ Running Drizzle ORM Tests..."
npx playwright test drizzle-orm.spec.ts --reporter=list

# 5. RAG Pipeline Integration Tests
echo ""
echo "5️⃣ Running RAG Pipeline Integration Tests..."
npx playwright test rag-pipeline-integration.spec.ts --reporter=list

# 6. GPU Acceleration Tests (optional)
echo ""
echo "6️⃣ Running GPU Acceleration Tests..."
npx playwright test gpu-acceleration.spec.ts --reporter=list || echo "⚠️  GPU tests skipped or failed (this is OK if no GPU is available)"

# Generate HTML report
echo ""
echo "📊 Generating test report..."
npx playwright show-report

echo ""
echo "✨ Test suite completed!"
echo "=================================================="