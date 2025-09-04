# Download and setup required services for Legal AI Platform
# This script downloads the large binaries instead of storing them in git

Write-Host "📦 Downloading Legal AI Platform Services..." -ForegroundColor Green

$downloads = @(
    @{
        Name = "MinIO"
        Url = "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
        Path = "minio.exe"
        Size = "~108MB"
    },
    @{
        Name = "Neo4j Community 5.23.0"
        Url = "https://dist.neo4j.org/neo4j-community-5.23.0-windows.zip"
        Path = "neo4j-community-5.23.0-windows.zip"
        Size = "~118MB"
    },
    @{
        Name = "Qdrant Vector Database"
        Url = "https://github.com/qdrant/qdrant/releases/download/v1.8.4/qdrant-x86_64-pc-windows-msvc.zip"
        Path = "qdrant-windows.zip"
        Size = "~77MB"
    }
)

foreach ($download in $downloads) {
    Write-Host "⬇️ Downloading $($download.Name) ($($download.Size))..." -ForegroundColor Cyan
    
    if (Test-Path $download.Path) {
        Write-Host "✅ $($download.Path) already exists, skipping..." -ForegroundColor Yellow
        continue
    }
    
    try {
        Invoke-WebRequest -Uri $download.Url -OutFile $download.Path -UseBasicParsing
        Write-Host "✅ Downloaded $($download.Name)" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download $($download.Name): $_" -ForegroundColor Red
    }
}

# Extract Qdrant if downloaded
if (Test-Path "qdrant-windows.zip") {
    Write-Host "📂 Extracting Qdrant..." -ForegroundColor Cyan
    Expand-Archive -Path "qdrant-windows.zip" -DestinationPath "qdrant-extracted" -Force
    Write-Host "✅ Qdrant extracted to qdrant-extracted/" -ForegroundColor Green
}

Write-Host "🎉 Service download complete!" -ForegroundColor Green
Write-Host "💡 These files are automatically downloaded and not stored in git." -ForegroundColor Yellow