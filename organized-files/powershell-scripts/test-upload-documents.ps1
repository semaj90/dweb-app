# Enhanced RAG Document Upload Test Script

Write-Host "🧪 Enhanced RAG Document Testing" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if documents exist
$uploadsPath = "uploads\documents"
if (Test-Path $uploadsPath) {
    Write-Host "📁 Found upload directory: $uploadsPath" -ForegroundColor Green
    $documents = Get-ChildItem $uploadsPath -Filter "*.md"
    Write-Host "   📄 Available documents: $($documents.Count)" -ForegroundColor Yellow

    foreach ($doc in $documents) {
        Write-Host "      - $($doc.Name) ($([math]::Round($doc.Length/1024, 1))KB)" -ForegroundColor Gray
    }
} else {
    Write-Host "❌ Upload directory not found" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test API endpoints
Write-Host "🔌 Testing API Endpoints..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173/api/rag?action=status" -Method GET -UseBasicParsing -TimeoutSec 5
    Write-Host "   ✅ RAG API Status: HTTP $($response.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ RAG API not responding" -ForegroundColor Red
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173/api/libraries" -Method GET -UseBasicParsing -TimeoutSec 5
    Write-Host "   ✅ Libraries API: HTTP $($response.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Libraries API not responding" -ForegroundColor Red
}

Write-Host ""

# Web interface instructions
Write-Host "🌐 How to Test Enhanced RAG:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Method 1 - Web Interface (Recommended):" -ForegroundColor Yellow
Write-Host "   1. Open: http://localhost:5173/rag-studio" -ForegroundColor White
Write-Host "   2. Use the upload button to upload documents" -ForegroundColor White
Write-Host "   3. Try queries like:" -ForegroundColor White
Write-Host "      • 'What are the legal frameworks?'" -ForegroundColor Gray
Write-Host "      • 'Explain the technical architecture'" -ForegroundColor Gray
Write-Host "      • 'What are the AI ethics principles?'" -ForegroundColor Gray

Write-Host ""
Write-Host "Method 2 - VS Code Extension:" -ForegroundColor Yellow
Write-Host "   1. Press: Ctrl+Shift+P" -ForegroundColor White
Write-Host "   2. Type: 'Context7 MCP: Enhanced RAG Query'" -ForegroundColor White
Write-Host "   3. Ask: 'Summarize the uploaded documents'" -ForegroundColor White

Write-Host ""
Write-Host "Method 3 - API Testing:" -ForegroundColor Yellow
Write-Host "   # Upload document via API" -ForegroundColor Gray
Write-Host "   curl -X POST 'http://localhost:5173/api/rag/upload' \" -ForegroundColor Gray
Write-Host "     -F 'file=@uploads/documents/test-legal-framework.md'" -ForegroundColor Gray
Write-Host ""
Write-Host "   # Query documents" -ForegroundColor Gray
Write-Host "   curl -X POST 'http://localhost:5173/api/rag/search' \" -ForegroundColor Gray
Write-Host "     -H 'Content-Type: application/json' \" -ForegroundColor Gray
Write-Host "     -d '{\"query\":\"legal requirements\",\"type\":\"semantic\"}'" -ForegroundColor Gray

Write-Host ""
Write-Host "📊 Test Documents Created:" -ForegroundColor Cyan
Write-Host "   📄 test-legal-framework.md - Legal compliance information" -ForegroundColor White
Write-Host "   📄 technical-manual.md - System architecture details" -ForegroundColor White
Write-Host "   📄 ai-ethics-policy.md - AI ethics and best practices" -ForegroundColor White

Write-Host ""
Write-Host "🎯 Expected Test Results:" -ForegroundColor Cyan
Write-Host "   ✅ Documents should be processed and indexed" -ForegroundColor Green
Write-Host "   ✅ Semantic search should find relevant content" -ForegroundColor Green
Write-Host "   ✅ Multi-agent workflows should provide detailed analysis" -ForegroundColor Green
Write-Host "   ✅ VS Code integration should work seamlessly" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 Ready to test! Open the web interface and start uploading!" -ForegroundColor Green
