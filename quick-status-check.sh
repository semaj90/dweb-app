#!/bin/bash
# Quick error check script

echo "🔍 Checking SvelteKit Frontend Errors..."
echo "================================================"

cd "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo "📁 Checking file structure..."
echo "- package.json exists: $(test -f package.json && echo "✅" || echo "❌")"
echo "- src/lib exists: $(test -d src/lib && echo "✅" || echo "❌")"
echo "- node_modules exists: $(test -d node_modules && echo "✅" || echo "❌")"

echo ""
echo "📦 Running TypeScript check..."
npx svelte-check --tsconfig ./tsconfig.json --output human 2>&1 | head -20

echo ""
echo "🏗️ Testing build..."
npm run build 2>&1 | head -10

echo ""
echo "✅ Quick check completed!"
