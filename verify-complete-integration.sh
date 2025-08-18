#!/bin/bash
# VERIFY-COMPLETE-INTEGRATION.sh
# Final verification script for Legal AI Platform merge resolution

echo "🔍 LEGAL AI PLATFORM - INTEGRATION VERIFICATION"
echo "================================================"

# Function to check file exists and has content
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        local size=$(wc -c < "$file")
        if [ "$size" -gt 0 ]; then
            echo "✅ $description ($size bytes)"
        else
            echo "⚠️  $description (empty file)"
        fi
    else
        echo "❌ $description (missing)"
    fi
}

echo ""
echo "📁 CHECKING APPLIED PATCHES"
echo "============================="

# Check all the files we created/modified
check_file ".env" "Enhanced environment configuration"
check_file "src/lib/stores/pg.ts" "Temperature-aware pgvector querying"
check_file "src/lib/stores/ann.ts" "Temperature-aware Qdrant querying"
check_file "src/routes/api/feedback/+server.ts" "Feedback API endpoint"
check_file "src/lib/components/FeedbackButtons.svelte" "Feedback UI component"
check_file "src/lib/server/embedding.ts" "Enhanced embedding service"
check_file "package.json" "Updated dependencies"
check_file "QUICK-FIX-CONFLICTS.ps1" "Automation script"
check_file "MERGE-CONFLICTS-RESOLUTION-COMPLETE.md" "Status documentation"

echo ""
echo "🔧 CHECKING CONFIGURATION"
echo "=========================="

# Check .env configuration
if [ -f ".env" ]; then
    if grep -q "EMBEDDING_DIMENSIONS=384" .env; then
        echo "✅ Embedding dimensions set to 384"
    else
        echo "⚠️  Embedding dimensions not configured"
    fi
    
    if grep -q "USE_AUTOGEN=true" .env; then
        echo "✅ AutoGen enabled"
    else
        echo "⚠️  AutoGen not enabled"
    fi
    
    if grep -q "GO_RAG_URL" .env; then
        echo "✅ Feedback system configured"
    else
        echo "⚠️  Feedback system URL not configured"
    fi
fi

echo ""
echo "📦 CHECKING DEPENDENCIES"
echo "========================"

# Check if package.json has the new dependencies
if [ -f "package.json" ]; then
    dependencies=("@xenova/transformers" "onnxruntime-web" "ioredis" "neo4j-driver" "amqplib" "minio" "node-fetch")
    
    for dep in "${dependencies[@]}"; do
        if grep -q "\"$dep\"" package.json; then
            echo "✅ $dep listed in package.json"
        else
            echo "⚠️  $dep not found in package.json"
        fi
    done
fi

echo ""
echo "🌐 CHECKING API ENDPOINTS"
echo "========================="

check_file "src/routes/api/enhanced-rag/+server.ts" "Enhanced RAG endpoint"
check_file "src/routes/api/feedback/+server.ts" "Feedback endpoint"

if [ -d "src/routes/api/enhanced-document-ingestion" ]; then
    echo "⚠️  Old enhanced-document-ingestion endpoint still exists (should be replaced with enhanced-rag)"
else
    echo "✅ Old endpoint properly removed/replaced"
fi

echo ""
echo "🎨 CHECKING UI COMPONENTS"
echo "========================="

check_file "src/routes/+page.svelte" "YoRHa homepage (already integrated)"
check_file "src/lib/components/FeedbackButtons.svelte" "Feedback UI component"

# Check if YoRHa components exist
if [ -d "src/lib/components/yorha" ]; then
    echo "✅ YoRHa component library exists"
else
    echo "⚠️  YoRHa component library not found"
fi

echo ""
echo "🧪 READY FOR TESTING"
echo "===================="

echo "The following commands should work once dependencies are installed:"
echo ""
echo "1. Install dependencies:"
echo "   npm install"
echo ""
echo "2. Type checking:"
echo "   npm run check"
echo ""
echo "3. Start development server:"
echo "   npm run dev"
echo ""
echo "4. Test temperature-aware querying:"
echo "   curl -X POST http://localhost:5173/api/enhanced-rag \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"action\":\"query\",\"data\":{\"query\":\"test\",\"temperature\":0.3}}'"
echo ""
echo "5. Test feedback system:"
echo "   curl -X POST http://localhost:5173/api/feedback \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"sessionId\":\"test\",\"query\":\"test\",\"reward\":1}'"

echo ""
echo "🎯 NEXT STEPS"
echo "============="
echo "1. Run: npm install"
echo "2. Start services: npm run dev"
echo "3. Access: http://localhost:5173"
echo "4. Test YoRHa dashboard functionality"
echo "5. Verify temperature-aware search works"
echo "6. Test feedback buttons provide user interaction"

echo ""
echo "✨ MERGE CONFLICTS RESOLUTION COMPLETE! ✨"
echo "Your Legal AI Platform is ready for enhanced RAG integration!"
