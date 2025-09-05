# Comprehensive System Verification Script - PowerShell Version
# Run with: powershell -ExecutionPolicy Bypass -File verify-system.ps1

Write-Host ""
Write-Host "🔍 COMPREHENSIVE SYSTEM VERIFICATION" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Initialize results
$results = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    services = @{}
    database = @{}
    apis = @{}
    processing = @{}
    recommendations = @()
}

# Function to check if port is open
function Test-Port {
    param(
        [string]$ComputerName = 'localhost',
        [int]$Port,
        [int]$Timeout = 3000
    )
    
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $connect = $tcpClient.BeginConnect($ComputerName, $Port, $null, $null)
        $wait = $connect.AsyncWaitHandle.WaitOne($Timeout, $false)
        
        if ($wait) {
            $tcpClient.EndConnect($connect)
            $tcpClient.Close()
            return $true
        } else {
            $tcpClient.Close()
            return $false
        }
    } catch {
        return $false
    }
}

# Function to check Windows service
function Test-WindowsService {
    param([string]$ServiceName)
    
    try {
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        return ($service.Status -eq 'Running')
    } catch {
        return $false
    }
}

# Function to check if process is running
function Test-Process {
    param([string]$ProcessName)
    
    $process = Get-Process -Name ($ProcessName -replace '\.exe$','') -ErrorAction SilentlyContinue
    return ($null -ne $process)
}

# Function to test HTTP endpoint
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = 'GET',
        [hashtable]$Body = @{}
    )
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            TimeoutSec = 5
            ErrorAction = 'Stop'
        }
        
        if ($Method -eq 'POST' -and $Body.Count -gt 0) {
            $params['Body'] = ($Body | ConvertTo-Json)
            $params['ContentType'] = 'application/json'
        }
        
        $response = Invoke-WebRequest @params
        
        return @{
            success = ($response.StatusCode -eq 200)
            status = $response.StatusCode
            data = $response.Content
        }
    } catch {
        return @{
            success = $false
            error = $_.Exception.Message
        }
    }
}

# 1. Check Core Services
Write-Host "📋 CHECKING CORE SERVICES" -ForegroundColor Blue
Write-Host "==========================" -ForegroundColor Blue

$services = @(
    @{Name='PostgreSQL'; Port=5432; ServiceName='postgresql-x64-15'; ProcessName='postgres'},
    @{Name='Redis'; Port=6379; ServiceName='Redis'; ProcessName='redis-server'},
    @{Name='Ollama'; Port=11434; ProcessName='ollama'},
    @{Name='Dev Server'; Port=5173}
)

foreach ($service in $services) {
    $portOpen = Test-Port -Port $service.Port
    $serviceRunning = $false
    $processRunning = $false
    
    if ($service.ServiceName) {
        $serviceRunning = Test-WindowsService -ServiceName $service.ServiceName
    }
    
    if ($service.ProcessName) {
        $processRunning = Test-Process -ProcessName $service.ProcessName
    }
    
    $isRunning = $portOpen -or $serviceRunning -or $processRunning
    
    $results.services[$service.Name] = @{
        running = $isRunning
        port = $service.Port
        portOpen = $portOpen
    }
    
    $statusSymbol = if ($isRunning) { "✅" } else { "❌" }
    $statusText = if ($isRunning) { "RUNNING" } else { "NOT RUNNING" }
    $statusColor = if ($isRunning) { "Green" } else { "Red" }
    $portStatus = if ($portOpen) { "✓" } else { "✗" }
    
    Write-Host "$($service.Name.PadRight(15)) | Port $($service.Port.ToString().PadRight(5)) [$portStatus] | " -NoNewline
    Write-Host "$statusSymbol $statusText" -ForegroundColor $statusColor
    
    if (-not $isRunning) {
        $results.recommendations += "Start $($service.Name) service"
    }
}

# 2. Database Connection Test
Write-Host ""
Write-Host "🗄️  DATABASE CONNECTION" -ForegroundColor Blue
Write-Host "========================" -ForegroundColor Blue

if ($results.services['PostgreSQL'].portOpen) {
    Write-Host "✅ PostgreSQL port is open" -ForegroundColor Green
    $results.database.accessible = $true
    
    # Test actual connection
    try {
        $env:PGPASSWORD = 'postgres'
        $output = & psql -U postgres -h localhost -p 5432 -c "SELECT version();" 2>$null
        if ($output) {
            Write-Host "✅ Database connection successful" -ForegroundColor Green
            $results.database.connectable = $true
        } else {
            Write-Host "⚠️  Could not verify database connection" -ForegroundColor Yellow
            $results.database.connectable = $false
        }
    } catch {
        Write-Host "⚠️  Could not verify database connection" -ForegroundColor Yellow
        $results.database.connectable = $false
    }
} else {
    Write-Host "❌ PostgreSQL not accessible" -ForegroundColor Red
    $results.database.accessible = $false
}

# 3. API Health Checks
Write-Host ""
Write-Host "🌐 API HEALTH CHECKS" -ForegroundColor Blue
Write-Host "======================" -ForegroundColor Blue

$apiEndpoints = @(
    @{Name='Dev Server Root'; Url='http://localhost:5173/'; Critical=$true},
    @{Name='OCR Service'; Url='http://localhost:5173/api/ocr/langextract'},
    @{Name='Embeddings Service'; Url='http://localhost:5173/api/embeddings/generate'},
    @{Name='Search Service'; Url='http://localhost:5173/api/documents/search'},
    @{Name='Storage Service'; Url='http://localhost:5173/api/documents/store'},
    @{Name='AI Upload Demo'; Url='http://localhost:5173/ai-upload-demo'}
)

foreach ($api in $apiEndpoints) {
    $result = Test-Endpoint -Url $api.Url
    $results.apis[$api.Name] = $result
    
    if ($result.success) {
        Write-Host "$($api.Name.PadRight(20)) | " -NoNewline
        Write-Host "✅ HEALTHY" -ForegroundColor Green
    } elseif ($result.error -like "*Unable to connect*") {
        Write-Host "$($api.Name.PadRight(20)) | " -NoNewline
        Write-Host "❌ UNREACHABLE" -ForegroundColor Red
        
        if ($api.Critical) {
            $results.recommendations += "Start the development server with: npm run dev"
        }
    } else {
        Write-Host "$($api.Name.PadRight(20)) | " -NoNewline
        Write-Host "⚠️  ERROR" -ForegroundColor Yellow
    }
    
    if ($result.error) {
        Write-Host "                       | $($result.error)" -ForegroundColor Red
    }
}

# 4. Ollama Model Check
Write-Host ""
Write-Host "🤖 AI MODEL STATUS" -ForegroundColor Blue
Write-Host "====================" -ForegroundColor Blue

if ($results.services['Ollama'].portOpen) {
    $ollamaTest = Test-Endpoint -Url 'http://localhost:11434/api/tags'
    
    if ($ollamaTest.success) {
        Write-Host "✅ Ollama API accessible" -ForegroundColor Green
        
        try {
            $modelsData = $ollamaTest.data | ConvertFrom-Json
            $models = $modelsData.models
            
            if ($models) {
                Write-Host "Found $($models.Count) model(s):"
                
                $requiredModels = @('nomic-embed-text', 'llama3.2')
                
                foreach ($requiredModel in $requiredModels) {
                    $hasModel = $models | Where-Object { $_.name -like "*$requiredModel*" }
                    
                    if ($hasModel) {
                        Write-Host "  ✓ $requiredModel" -ForegroundColor Green
                    } else {
                        Write-Host "  ✗ $requiredModel - NOT INSTALLED" -ForegroundColor Yellow
                        $results.recommendations += "Install model: ollama pull $requiredModel"
                    }
                }
            }
        } catch {
            Write-Host "Could not parse Ollama models" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Ollama API not responding" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Ollama not running" -ForegroundColor Red
    $results.recommendations += "Start Ollama service"
}

# 5. Test Real Processing
if ($results.apis['Dev Server Root'].success) {
    Write-Host ""
    Write-Host "🧪 TESTING REAL PROCESSING" -ForegroundColor Blue
    Write-Host "============================" -ForegroundColor Blue
    
    $embeddingTest = Test-Endpoint -Url 'http://localhost:5173/api/embeddings/generate' `
                                   -Method 'POST' `
                                   -Body @{
                                       text = 'Test legal contract document verification'
                                       model = 'nomic-embed-text'
                                   }
    
    if ($embeddingTest.success) {
        Write-Host "✅ Real embedding generation working" -ForegroundColor Green
        $results.processing.embeddings = $true
    } else {
        Write-Host "❌ Embedding generation failed" -ForegroundColor Red
        $results.processing.embeddings = $false
    }
}

# 6. System Summary
Write-Host ""
Write-Host "📊 VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

$serviceCount = ($results.services.Values | Where-Object { $_.running }).Count
$apiCount = ($results.apis.Values | Where-Object { $_.success }).Count
$totalServices = $results.services.Count
$totalApis = $results.apis.Count

Write-Host "Services Running: $serviceCount/$totalServices"
Write-Host "APIs Healthy: $apiCount/$totalApis"
Write-Host "Database: $(if ($results.database.accessible) { 'Accessible' } else { 'Not Accessible' })"
Write-Host "Processing: $(if ($results.processing.embeddings) { 'Working' } else { 'Not Working' })"

$allGood = ($serviceCount -eq $totalServices) -and `
           ($apiCount -eq $totalApis) -and `
           $results.database.accessible -and `
           $results.processing.embeddings

if ($allGood) {
    Write-Host ""
    Write-Host "🎉 SYSTEM FULLY OPERATIONAL!" -ForegroundColor Green
    Write-Host "✅ All services running" -ForegroundColor Green
    Write-Host "✅ All APIs healthy" -ForegroundColor Green
    Write-Host "✅ Database accessible" -ForegroundColor Green
    Write-Host "✅ Real processing working" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔗 Ready to use at: " -NoNewline -ForegroundColor Magenta
    Write-Host "http://localhost:5173/ai-upload-demo" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "⚠️  SYSTEM PARTIALLY OPERATIONAL" -ForegroundColor Yellow
    
    if ($results.recommendations.Count -gt 0) {
        Write-Host ""
        Write-Host "💡 RECOMMENDED ACTIONS:" -ForegroundColor Yellow
        $i = 1
        foreach ($rec in $results.recommendations | Select-Object -Unique) {
            Write-Host "$i. $rec"
            $i++
        }
    }
    
    Write-Host ""
    Write-Host "🔧 QUICK START COMMANDS:" -ForegroundColor Blue
    Write-Host "1. Start all services: .\START-REAL-SYSTEM.bat"
    Write-Host "2. Start dev server: npm run dev"
    Write-Host "3. Check Ollama: ollama list"
    Write-Host '4. Test database: psql -U postgres -c "\l"'
}

# Save results
Write-Host ""
Write-Host "💾 Saving detailed results..." -ForegroundColor Cyan
$results | ConvertTo-Json -Depth 5 | Out-File -FilePath "verification-results.json"
Write-Host "Results saved to: verification-results.json"

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
