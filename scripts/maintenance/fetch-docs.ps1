# Enhanced Documentation Fetcher for Legal AI Web-App
Write-Host "📥 Fetching documentation for legal AI web-app..." -ForegroundColor Cyan

# Create directories
New-Item -ItemType Directory -Force -Path "docs\raw" | Out-Null

$urls = @(
    # Core web technologies
    "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/index.html",
    "https://raw.githubusercontent.com/mdn/content/main/files/en-us/webassembly/index.html",
    
    # UI Framework documentation  
    "https://bits-ui.com/docs/getting-started",
    "https://bits-ui.com/docs/components/dialog",
    "https://bits-ui.com/docs/components/button",
    "https://bits-ui.com/docs/components/form",
    "https://next.melt-ui.com/guides/how-to-use",
    
    # Styling and CSS
    "https://tailwindcss.com/docs/installation",
    "https://tailwindcss.com/docs/responsive-design",
    
    # Database and ORM
    "https://orm.drizzle.team/docs/guides",
    "https://orm.drizzle.team/docs/kit-overview",
    "https://www.postgresql.org/docs/current/tutorial.html",
    
    # AI/ML and LLM documentation
    "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/README.md",
    "https://docs.anthropic.com/claude/docs/intro-to-claude",
    
    # SvelteKit framework
    "https://kit.svelte.dev/docs/introduction",
    "https://kit.svelte.dev/docs/routing",
    "https://svelte.dev/docs/introduction",
    
    # TypeScript for legal applications
    "https://www.typescriptlang.org/docs/handbook/basic-types.html",
    "https://www.typescriptlang.org/docs/handbook/interfaces.html"
)

$totalUrls = $urls.Count
$current = 1
$successCount = 0

Write-Host "🚀 Starting download of $totalUrls documentation sources..." -ForegroundColor Green

foreach ($url in $urls) {
    $percentComplete = [math]::Round(($current / $totalUrls) * 100, 1)
    Write-Progress -Activity "Fetching Documentation" -Status "Processing $url" -PercentComplete $percentComplete
    
    $fileName = $url -replace '[:\\/]+', '_'
    $outputPath = "docs\raw\$fileName.html"
    
    try {
        # Add user agent to avoid blocking
        $headers = @{
            'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        Invoke-WebRequest -Uri $url -OutFile $outputPath -UseBasicParsing -TimeoutSec 30 -Headers $headers
        Write-Host "✅ ($current/$totalUrls) Fetched: $([System.IO.Path]::GetFileName($outputPath))" -ForegroundColor Green
        $successCount++
    }
    catch {
        Write-Host "❌ ($current/$totalUrls) Failed: $([System.IO.Path]::GetFileName($outputPath))" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
    }
    
    Start-Sleep -Milliseconds 500
    $current++
}

Write-Progress -Activity "Fetching Documentation" -Completed

Write-Host "`n🎉 Documentation fetch complete!" -ForegroundColor Green
Write-Host "📊 Successfully downloaded: $successCount out of $totalUrls files" -ForegroundColor Yellow
Write-Host "📁 Files saved to: docs\raw\" -ForegroundColor Gray

if ($successCount -eq 0) {
    Write-Host "⚠️  No files were downloaded. Check your internet connection." -ForegroundColor Red
} elseif ($successCount -lt $totalUrls) {
    Write-Host "⚠️  Some files failed to download. This is normal for some sites." -ForegroundColor Yellow
} else {
    Write-Host "🏆 Perfect! All documentation downloaded successfully." -ForegroundColor Green
}
