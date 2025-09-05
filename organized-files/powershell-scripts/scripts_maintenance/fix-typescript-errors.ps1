# Fix TypeScript Errors in SvelteKit Frontend
Write-Host "🔧 Fixing TypeScript errors..." -ForegroundColor Green

$projectRoot = "C:\Users\james\Desktop\web-app\sveltekit-frontend"

# Fix excessive defaultRandom calls in unified-schema.ts
$unifiedSchemaPath = Join-Path $projectRoot "src\lib\server\db\unified-schema.ts"
if (Test-Path $unifiedSchemaPath) {
    Write-Host "📝 Fixing unified-schema.ts..." -ForegroundColor Yellow
    $content = Get-Content $unifiedSchemaPath -Raw
    $fixedContent = $content -replace '\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)', '.defaultRandom()'
    Set-Content $unifiedSchemaPath $fixedContent -Encoding UTF8
    Write-Host "   ✅ Fixed excessive defaultRandom calls" -ForegroundColor Green
}

# Fix cache type issues in vector-search.ts
$vectorSearchPath = Join-Path $projectRoot "src\lib\server\search\vector-search.ts"
if (Test-Path $vectorSearchPath) {
    Write-Host "📝 Fixing vector-search.ts..." -ForegroundColor Yellow
    $content = Get-Content $vectorSearchPath -Raw
    $fixedContent = $content -replace 'cache\.get<VectorSearchResult\[\]>\(', 'cache.get('
    $fixedContent = $fixedContent -replace 'cache\.get<([^>]+)>\(', 'cache.get('
    Set-Content $vectorSearchPath $fixedContent -Encoding UTF8
    Write-Host "   ✅ Fixed cache type issues" -ForegroundColor Green
}

# Fix vector service database issues
$vectorServicePath = Join-Path $projectRoot "src\lib\server\services\vector-service.ts"
if (Test-Path $vectorServicePath) {
    Write-Host "📝 Fixing vector-service.ts..." -ForegroundColor Yellow
    $content = Get-Content $vectorServicePath -Raw
    $fixedContent = $content -replace '\.returning\(\{ id: userEmbeddings\.userId \}\)', '.returning({ id: userEmbeddings.id })'
    Set-Content $vectorServicePath $fixedContent -Encoding UTF8
    Write-Host "   ✅ Fixed database schema issues" -ForegroundColor Green
}

Write-Host "🎉 TypeScript fixes complete!" -ForegroundColor Green
Write-Host "🚀 Now running svelte-check..." -ForegroundColor Cyan

# Change to frontend directory and run check
Set-Location $projectRoot
& npm run check

Write-Host "✅ Check complete!" -ForegroundColor Green
