# Smart Cache Manager - PyTorch-style caching for AI models and dependencies
# Checks for existing installations and downloads only if needed

param(
    [switch]$Force,
    [switch]$Verbose,
    [string]$CacheDir = "$env:USERPROFILE\.legal-ai-cache"
)

$ErrorActionPreference = "Continue"

# Colors for output
function Write-Success { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "ℹ️  $msg" -ForegroundColor Cyan }
function Write-Warning { param($msg) Write-Host "⚠️  $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }
function Write-Header { 
    param($msg) 
    Write-Host "`n$('='*60)" -ForegroundColor Blue
    Write-Host $msg -ForegroundColor Blue
    Write-Host "$('='*60)`n" -ForegroundColor Blue
}

function Write-Cache { param($msg) Write-Host "📦 $msg" -ForegroundColor Magenta }

Write-Header "LEGAL AI CACHE MANAGER"

# Create cache directory structure
$CacheDirs = @(
    "$CacheDir\models",
    "$CacheDir\binaries", 
    "$CacheDir\temp",
    "$CacheDir\logs"
)

foreach ($dir in $CacheDirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Cache "Created cache directory: $dir"
    }
}

# Cache manifest file
$ManifestFile = "$CacheDir\manifest.json"
$Manifest = @{}

if (Test-Path $ManifestFile) {
    try {
        $Manifest = Get-Content $ManifestFile | ConvertFrom-Json -AsHashtable
        Write-Cache "Loaded existing cache manifest"
    } catch {
        Write-Warning "Invalid manifest file, creating new one"
        $Manifest = @{}
    }
} else {
    Write-Cache "Creating new cache manifest"
}

# Function to check if item is cached and valid
function Test-CacheItem {
    param($Name, $Version, $Type = "binary")
    
    $key = "$Name-$Version"
    if ($Manifest.ContainsKey($key)) {
        $item = $Manifest[$key]
        $path = $item.path
        
        if (Test-Path $path) {
            # Check file hash if available
            if ($item.hash) {
                $currentHash = Get-FileHash $path -Algorithm SHA256
                if ($currentHash.Hash -eq $item.hash) {
                    Write-Cache "✅ $Name v$Version found in cache (verified)"
                    return $true
                } else {
                    Write-Warning "Hash mismatch for $Name, will re-download"
                    return $false
                }
            } else {
                Write-Cache "✅ $Name v$Version found in cache"
                return $true
            }
        }
    }
    
    Write-Cache "❌ $Name v$Version not in cache or invalid"
    return $false
}

# Function to add item to cache
function Add-CacheItem {
    param($Name, $Version, $Path, $Type = "binary", $Hash = $null)
    
    $key = "$Name-$Version"
    $Manifest[$key] = @{
        name = $Name
        version = $Version
        type = $Type
        path = $Path
        hash = $Hash
        cached_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    
    # Save manifest
    $Manifest | ConvertTo-Json -Depth 3 | Set-Content $ManifestFile
    Write-Cache "📝 Added $Name v$Version to cache manifest"
}

# Function to download with progress
function Download-WithProgress {
    param($Url, $OutFile, $Description)
    
    Write-Info "Downloading $Description..."
    
    try {
        # Use .NET WebClient for progress reporting
        $webClient = New-Object System.Net.WebClient
        
        # Progress event handler
        $webClient.add_DownloadProgressChanged({
            param($sender, $e)
            $percent = $e.ProgressPercentage
            $received = [math]::Round($e.BytesReceived / 1MB, 2)
            $total = [math]::Round($e.TotalBytesToReceive / 1MB, 2)
            
            Write-Progress -Activity "Downloading $Description" -Status "$received MB / $total MB" -PercentComplete $percent
        })
        
        # Download file
        $webClient.DownloadFileTaskAsync($Url, $OutFile).Wait()
        $webClient.Dispose()
        
        Write-Progress -Activity "Downloading $Description" -Completed
        Write-Success "Downloaded $Description"
        return $true
        
    } catch {
        Write-Error "Failed to download $Description : $_"
        return $false
    }
}

Write-Header "CHECKING DEPENDENCIES"

# 1. Check MinIO
Write-Info "Checking MinIO installation..."
$MinIOVersion = "RELEASE.2024-01-16T16-07-38Z"
$MinIOPath = "$CacheDir\binaries\minio.exe"

if (-not (Test-CacheItem "minio" $MinIOVersion "binary") -or $Force) {
    Write-Info "Downloading MinIO $MinIOVersion..."
    $MinIOUrl = "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
    
    if (Download-WithProgress $MinIOUrl $MinIOPath "MinIO Server") {
        $hash = (Get-FileHash $MinIOPath -Algorithm SHA256).Hash
        Add-CacheItem "minio" $MinIOVersion $MinIOPath "binary" $hash
        
        # Add to PATH for current session
        $env:PATH += ";$CacheDir\binaries"
        Write-Success "MinIO cached and ready"
    }
} else {
    # Add to PATH
    $env:PATH += ";$CacheDir\binaries"
    Write-Success "MinIO already cached"
}

# 2. Check Ollama Models
Write-Info "Checking Ollama models..."

$RequiredModels = @(
    @{ name = "nomic-embed-text"; size = "274MB"; description = "Text embeddings" },
    @{ name = "llama3.2"; size = "2.0GB"; description = "Language model" },
    @{ name = "gemma2:2b"; size = "1.6GB"; description = "Lightweight LLM" }
)

# Check if Ollama is running
$ollamaRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction Stop
    $ollamaRunning = $true
    Write-Success "Ollama service is running"
} catch {
    Write-Warning "Ollama service not running, will start it"
}

if ($ollamaRunning) {
    foreach ($model in $RequiredModels) {
        $modelName = $model.name
        Write-Info "Checking model: $modelName"
        
        # Check if model exists in Ollama
        try {
            $models = ollama list | Out-String
            if ($models -match $modelName) {
                Write-Success "Model $modelName already available"
                Add-CacheItem "ollama-model-$modelName" "latest" "ollama://local" "model"
            } else {
                if (-not $Force) {
                    Write-Info "Model $modelName not found. Size: $($model.size)"
                    $download = Read-Host "Download $modelName ($($model.description))? [Y/n]"
                    if ($download -eq "" -or $download -eq "Y" -or $download -eq "y") {
                        Write-Info "Pulling model $modelName (this may take a while)..."
                        ollama pull $modelName
                        if ($LASTEXITCODE -eq 0) {
                            Write-Success "Model $modelName downloaded successfully"
                            Add-CacheItem "ollama-model-$modelName" "latest" "ollama://local" "model"
                        } else {
                            Write-Error "Failed to download model $modelName"
                        }
                    }
                } else {
                    Write-Info "Force downloading model $modelName..."
                    ollama pull $modelName
                }
            }
        } catch {
            Write-Warning "Could not check model $modelName : $_"
        }
    }
}

# 3. Check Go Dependencies
Write-Info "Checking Go module cache..."
$GoModCacheDir = "$CacheDir\go-mod-cache"
if (!(Test-Path $GoModCacheDir)) {
    New-Item -ItemType Directory -Path $GoModCacheDir -Force | Out-Null
}

# Set Go module cache environment
$env:GOMODCACHE = $GoModCacheDir
Write-Info "Go module cache set to: $GoModCacheDir"

# 4. Check Node.js Dependencies
Write-Info "Checking Node.js package cache..."
$NodeCacheDir = "$CacheDir\npm-cache"
if (!(Test-Path $NodeCacheDir)) {
    New-Item -ItemType Directory -Path $NodeCacheDir -Force | Out-Null
}

# Set npm cache
npm config set cache $NodeCacheDir
Write-Info "npm cache set to: $NodeCacheDir"

# 5. Check PostgreSQL Extensions
Write-Info "Checking PostgreSQL extensions..."
try {
    $pgResult = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT * FROM pg_extension WHERE extname = 'vector';" -t 2>$null
    if ($pgResult -match "vector") {
        Write-Success "PostgreSQL pgvector extension is installed"
        Add-CacheItem "pgvector" "latest" "postgresql://extension" "extension"
    } else {
        Write-Warning "pgvector extension not found"
        Write-Info "Installing pgvector extension..."
        & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "pgvector extension installed"
            Add-CacheItem "pgvector" "latest" "postgresql://extension" "extension"
        } else {
            Write-Error "Failed to install pgvector extension"
        }
    }
} catch {
    Write-Warning "Could not check PostgreSQL extensions: $_"
}

# 6. Precompile SvelteKit
Write-Info "Checking SvelteKit build cache..."
$SvelteKitCacheDir = "$CacheDir\sveltekit-cache"
if (!(Test-Path $SvelteKitCacheDir)) {
    New-Item -ItemType Directory -Path $SvelteKitCacheDir -Force | Out-Null
}

# Create cache summary
Write-Header "CACHE SUMMARY"

$cacheSize = 0
Get-ChildItem $CacheDir -Recurse -File | ForEach-Object { $cacheSize += $_.Length }
$cacheSizeMB = [math]::Round($cacheSize / 1MB, 2)

Write-Info "Cache location: $CacheDir"
Write-Info "Cache size: $cacheSizeMB MB"
Write-Info "Cached items: $($Manifest.Count)"

if ($Verbose) {
    Write-Info "`nCached items:"
    foreach ($item in $Manifest.GetEnumerator()) {
        $details = $item.Value
        Write-Host "  • $($details.name) v$($details.version) ($($details.type))" -ForegroundColor Gray
    }
}

# Export cache paths for other scripts
$CacheConfig = @{
    cache_dir = $CacheDir
    minio_path = $MinIOPath
    go_mod_cache = $GoModCacheDir
    npm_cache = $NodeCacheDir
    sveltekit_cache = $SvelteKitCacheDir
}

$CacheConfig | ConvertTo-Json | Set-Content "$CacheDir\config.json"

Write-Success "Cache check complete!"
Write-Info "Use START-MINIO-INTEGRATION.bat to start services with cached dependencies"

# Return cache status
exit 0