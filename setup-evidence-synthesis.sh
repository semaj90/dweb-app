#!/bin/bash
# Quick Production Setup Script
# File: setup-evidence-synthesis.sh

set -e

echo "🚀 Setting up Evidence Synthesis Workflow"

# Check Node.js dependencies
echo "📦 Installing dependencies..."
cd sveltekit-frontend
npm install node-fetch redis

# Environment validation
echo "🔧 Validating environment..."
if [ ! -f .env ]; then
    echo "⚠️  Creating .env from template"
    cp .env.example .env
fi

# Database setup check
echo "🗄️  Checking database connection..."
npm run db:generate 2>/dev/null || echo "⚠️  Run: npm run db:generate"

# Redis setup check
echo "📡 Checking Redis connection..."
redis-cli ping 2>/dev/null || echo "⚠️  Start Redis: redis-server"

# Test AI services
echo "🧠 Validating AI services..."
curl -s http://localhost:11434/api/tags >/dev/null || echo "⚠️  Start Ollama: ollama serve"

echo "✅ Setup complete. Run workflow test:"
echo "   node ../test-evidence-synthesis-workflow.mjs"
