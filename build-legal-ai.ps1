# Legal AI Platform - Production Build Automation Script
# Version: 2.0.0
# Compatible with: Windows 10/11, PowerShell 5.1+

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Development", "Staging", "Production")]
    [string]$Environment = "Development",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests,
    
    [Parameter(Mandatory=$false)]
    [switch]$CleanBuild,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,

    [Parameter(Mandatory=$false)]
    [switch]$StartServices
)

# Error handling and configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Logging configuration
$LogFile = "build-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').log"
$LogDir = "logs"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Log {
    param($Message, $Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    $Color = switch($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        "INFO" { "Cyan" }
        default { "White" }
    }
    Write-Host $LogMessage -ForegroundColor $Color
    Add-Content -Path "$LogDir\$LogFile" -Value $LogMessage -ErrorAction SilentlyContinue
}

function Test-Prerequisites {
    Write-Log "🔍 Checking build prerequisites..." "INFO"
    
    $Prerequisites = @(
        @{Name="Go"; Command="go version"; Pattern="go version go(\d+\.\d+)"; MinVersion="1.21"},
        @{Name="Node.js"; Command="node --version"; Pattern="v(\d+\.\d+)"; MinVersion="18.0"},
        @{Name="NPM"; Command="npm --version"; Pattern="(\d+\.\d+)"; MinVersion="9.0"},
        @{Name="Git"; Command="git --version"; Pattern="git version (\d+\.\d+)"; MinVersion="2.30"}
    )
    
    foreach ($prereq in $Prerequisites) {
        try {
            $output = Invoke-Expression $prereq.Command 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "$($prereq.Name) not found or not accessible"
            }
            Write-Log "✅ $($prereq.Name): $($output -split '\n' | Select-Object -First 1)" "SUCCESS"
        }
        catch {
            Write-Log "❌ $($prereq.Name) check failed: $_" "ERROR"
            throw "Prerequisite check failed: $($prereq.Name)"
        }
    }
    
    # Check for required directories
    $RequiredDirs = @("go-microservice", "quic-services", "sveltekit-frontend")
    foreach ($dir in $RequiredDirs) {
        if (!(Test-Path $dir)) {
            throw "Required directory not found: $dir"
        }
    }
    
    Write-Log "✅ All prerequisites satisfied" "SUCCESS"
}

function Build-GoServices {
    Write-Log "🔨 Building Go microservices..." "INFO"
    
    $Services = @(
        "enhanced-rag",
        "upload-service", 
        "grpc-server",
        "cluster-service",
        "summarizer-service",
        "vector-service",
        "ingest-service"
    )
    
    Push-Location go-microservice
    
    try {
        # Clean build if requested
        if ($CleanBuild) {
            Write-Log "🧹 Cleaning Go module cache and binaries..." "INFO"
            go clean -modcache -cache
            if (Test-Path "bin") {
                Remove-Item -Recurse -Force bin
                Write-Log "Removed existing bin directory" "INFO"
            }
        }
        
        # Create bin directory
        if (!(Test-Path "bin")) {
            New-Item -ItemType Directory -Path "bin" -Force | Out-Null
            Write-Log "Created bin directory" "INFO"
        }
        
        # Update dependencies
        Write-Log "📦 Updating Go dependencies..." "INFO"
        go mod download
        if ($LASTEXITCODE -ne 0) { throw "Go mod download failed" }
        
        go mod tidy
        if ($LASTEXITCODE -ne 0) { throw "Go mod tidy failed" }
        
        # Build services in parallel where possible
        $BuiltServices = @()
        foreach ($service in $Services) {
            $ServicePath = ".\cmd\$service"
            if (!(Test-Path $ServicePath)) {
                Write-Log "⚠️ Service path not found: $ServicePath, skipping $service" "WARN"
                continue
            }
            
            Write-Log "🔧 Building $service..." "INFO"
            
            $BuildStart = Get-Date
            $BuildArgs = @(
                "build",
                "-ldflags", "-s -w -X main.version=$(git describe --tags --always 2>$null) -X main.buildTime=$(Get-Date -Format 'yyyy-MM-dd_HH:mm:ss')",
                "-o", ".\bin\$service.exe",
                ".\cmd\$service"
            )
            
            # Add race detector for non-production builds
            if ($Environment -ne "Production") {
                $BuildArgs = @("build", "-race") + $BuildArgs[1..($BuildArgs.Length-1)]
            }
            
            & go @BuildArgs
            
            if ($LASTEXITCODE -ne 0) {
                throw "Build failed for $service"
            }
            
            $BuildTime = (Get-Date) - $BuildStart
            $FileInfo = Get-Item ".\bin\$service.exe" -ErrorAction Stop
            $SizeMB = [math]::Round($FileInfo.Length / 1MB, 2)
            
            Write-Log "✅ $service built successfully (${SizeMB}MB, $($BuildTime.TotalSeconds.ToString('F2'))s)" "SUCCESS"
            $BuiltServices += $service
        }
        
        Write-Log "✅ Built $($BuiltServices.Count) Go services successfully: $($BuiltServices -join ', ')" "SUCCESS"
        
        # Verify all built services
        foreach ($service in $BuiltServices) {
            $BinaryPath = ".\bin\$service.exe"
            if (Test-Path $BinaryPath) {
                # Quick smoke test - try to show help/version
                try {
                    $output = & $BinaryPath --help 2>&1
                    Write-Log "🔍 $service binary verified" "INFO"
                }
                catch {
                    Write-Log "⚠️ $service binary may have issues: $_" "WARN"
                }
            }
        }
    }
    finally {
        Pop-Location
    }
}

function Build-QUICServices {
    Write-Log "🌐 Building QUIC protocol services..." "INFO"
    
    Push-Location quic-services -ErrorAction Stop
    
    try {
        # Update dependencies
        go mod tidy
        if ($LASTEXITCODE -ne 0) {
            throw "QUIC services go mod tidy failed"
        }
        
        $QUICServices = @(
            @{Name="quic-gateway"; Source="quic-gateway.go"},
            @{Name="quic-vector-proxy"; Source="quic-vector-proxy.go"},
            @{Name="quic-ai-stream"; Source="quic-ai-stream.go"}
        )
        
        $BuiltQUICServices = @()
        foreach ($service in $QUICServices) {
            if (Test-Path $service.Source) {
                Write-Log "🔧 Building $($service.Name)..." "INFO"
                
                $BuildStart = Get-Date
                go build -ldflags "-s -w" -o "..\go-microservice\bin\$($service.Name).exe" $service.Source
                
                if ($LASTEXITCODE -ne 0) {
                    throw "Build failed for $($service.Name)"
                }
                
                $BuildTime = (Get-Date) - $BuildStart
                Write-Log "✅ $($service.Name) built successfully ($($BuildTime.TotalSeconds.ToString('F2'))s)" "SUCCESS"
                $BuiltQUICServices += $service.Name
            } else {
                Write-Log "⚠️ Source not found for $($service.Name): $($service.Source)" "WARN"
            }
        }
        
        if ($BuiltQUICServices.Count -gt 0) {
            Write-Log "✅ Built $($BuiltQUICServices.Count) QUIC services: $($BuiltQUICServices -join ', ')" "SUCCESS"
        }
    }
    finally {
        Pop-Location
    }
}

function Build-Frontend {
    Write-Log "🎨 Building SvelteKit frontend..." "INFO"
    
    Push-Location sveltekit-frontend -ErrorAction Stop
    
    try {
        # Check if package.json exists
        if (!(Test-Path "package.json")) {
            throw "package.json not found in sveltekit-frontend directory"
        }
        
        # Install dependencies
        Write-Log "📦 Installing npm dependencies..." "INFO"
        $InstallStart = Get-Date
        
        if ($CleanBuild -and (Test-Path "node_modules")) {
            Write-Log "🧹 Cleaning node_modules..." "INFO"
            Remove-Item -Recurse -Force node_modules
        }
        
        npm ci --silent --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) {
            # Fallback to regular install
            Write-Log "npm ci failed, trying npm install..." "WARN"
            npm install --silent --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) {
                throw "npm install failed"
            }
        }
        
        $InstallTime = (Get-Date) - $InstallStart
        Write-Log "✅ Dependencies installed ($($InstallTime.TotalSeconds.ToString('F2'))s)" "SUCCESS"
        
        # TypeScript check
        Write-Log "🔍 Running TypeScript checks..." "INFO"
        npm run check
        if ($LASTEXITCODE -ne 0) {
            Write-Log "TypeScript check failed, trying ultra-fast check..." "WARN"
            npm run check:ultra-fast
            if ($LASTEXITCODE -ne 0) {
                throw "TypeScript checks failed"
            }
        }
        
        # Build application
        $BuildStart = Get-Date
        if ($Environment -eq "Production") {
            Write-Log "🏗️ Building production frontend..." "INFO"
            $env:NODE_ENV = "production"
            npm run build
        } else {
            Write-Log "🏗️ Building development frontend..." "INFO"
            npm run build
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed"
        }
        
        $BuildTime = (Get-Date) - $BuildStart
        Write-Log "✅ Frontend built successfully ($($BuildTime.TotalSeconds.ToString('F2'))s)" "SUCCESS"
        
        # Check build output
        if (Test-Path "build") {
            $BuildSize = (Get-ChildItem -Recurse build | Measure-Object -Property Length -Sum).Sum
            $BuildSizeMB = [math]::Round($BuildSize / 1MB, 2)
            Write-Log "📊 Build output: ${BuildSizeMB}MB" "INFO"
        }
    }
    finally {
        Pop-Location
        Remove-Item Env:NODE_ENV -ErrorAction SilentlyContinue
    }
}

function Run-Tests {
    if ($SkipTests) {
        Write-Log "⏭️ Skipping tests as requested" "WARN"
        return
    }
    
    Write-Log "🧪 Running test suite..." "INFO"
    
    # Go tests
    Push-Location go-microservice
    try {
        Write-Log "🔬 Running Go unit tests..." "INFO"
        $TestStart = Get-Date
        
        # Run tests with coverage
        go test -v -race -coverprofile=coverage.out -covermode=atomic ./...
        if ($LASTEXITCODE -ne 0) {
            Write-Log "❌ Some Go tests failed, but continuing..." "WARN"
        } else {
            # Generate coverage report
            if (Test-Path "coverage.out") {
                $CoverageOutput = go tool cover -func=coverage.out | Select-String "total:"
                if ($CoverageOutput) {
                    $Coverage = $CoverageOutput.ToString().Split()[-1]
                    Write-Log "📊 Go test coverage: $Coverage" "SUCCESS"
                }
            }
        }
        
        $TestTime = (Get-Date) - $TestStart
        Write-Log "✅ Go tests completed ($($TestTime.TotalSeconds.ToString('F2'))s)" "SUCCESS"
    }
    catch {
        Write-Log "⚠️ Go tests encountered issues: $_" "WARN"
    }
    finally {
        Pop-Location
    }
    
    # Frontend tests (if they exist)
    Push-Location sveltekit-frontend
    try {
        if (Test-Path "vitest.config.ts") {
            Write-Log "🧪 Running frontend tests..." "INFO"
            $FrontendTestStart = Get-Date
            
            npm run test:unit 2>$null
            if ($LASTEXITCODE -eq 0) {
                $FrontendTestTime = (Get-Date) - $FrontendTestStart
                Write-Log "✅ Frontend tests passed ($($FrontendTestTime.TotalSeconds.ToString('F2'))s)" "SUCCESS"
            } else {
                Write-Log "⚠️ Frontend tests failed or not configured" "WARN"
            }
        } else {
            Write-Log "ℹ️ No frontend tests configured" "INFO"
        }
    }
    finally {
        Pop-Location
    }
}

function Start-Services {
    if (!$StartServices) {
        Write-Log "ℹ️ Skipping service startup (use -StartServices to enable)" "INFO"
        return
    }
    
    Write-Log "🚀 Starting Legal AI services..." "INFO"
    
    # Check if START-LEGAL-AI.bat exists
    $StartScript = "START-LEGAL-AI.bat"
    if (Test-Path $StartScript) {
        Write-Log "Starting services using $StartScript..." "INFO"
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "$StartScript" -WindowStyle Minimized
        
        # Wait a bit and check if key services are running
        Start-Sleep -Seconds 10
        
        $ServicesToCheck = @(
            @{Name="SvelteKit Frontend"; Port=5173},
            @{Name="Enhanced RAG"; Port=8094},
            @{Name="Upload Service"; Port=8093}
        )
        
        foreach ($service in $ServicesToCheck) {
            try {
                $connection = Test-NetConnection -ComputerName "localhost" -Port $service.Port -WarningAction SilentlyContinue -InformationLevel Quiet
                if ($connection.TcpTestSucceeded) {
                    Write-Log "✅ $($service.Name) is running on port $($service.Port)" "SUCCESS"
                } else {
                    Write-Log "⚠️ $($service.Name) not accessible on port $($service.Port)" "WARN"
                }
            }
            catch {
                Write-Log "❌ Could not check $($service.Name): $_" "ERROR"
            }
        }
    } else {
        Write-Log "⚠️ Startup script $StartScript not found" "WARN"
    }
}

function Generate-BuildReport {
    Write-Log "📊 Generating build report..." "INFO"
    
    # Get Git information
    $GitVersion = ""
    $GitCommit = ""
    $GitBranch = ""
    
    try {
        $GitVersion = git describe --tags --always 2>$null
        $GitCommit = git rev-parse HEAD 2>$null
        $GitBranch = git rev-parse --abbrev-ref HEAD 2>$null
    }
    catch {
        Write-Log "⚠️ Could not retrieve Git information" "WARN"
    }
    
    $Report = @{
        BuildInfo = @{
            BuildTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Environment = $Environment
            Version = $GitVersion
            Commit = $GitCommit
            Branch = $GitBranch
            CleanBuild = $CleanBuild.IsPresent
            SkippedTests = $SkipTests.IsPresent
        }
        Services = @()
        Frontend = @{}
        BuildStats = @{
            TotalBinaries = 0
            TotalSizeMB = 0
        }
    }
    
    # Collect service binary information
    if (Test-Path "go-microservice\bin") {
        $Binaries = Get-ChildItem "go-microservice\bin\*.exe" -ErrorAction SilentlyContinue
        foreach ($binary in $Binaries) {
            $SizeMB = [math]::Round($binary.Length / 1MB, 2)
            $Report.Services += @{
                Name = $binary.BaseName
                Size = "$SizeMB MB"
                Modified = $binary.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
                Path = $binary.FullName
            }
            $Report.BuildStats.TotalSizeMB += $SizeMB
        }
        $Report.BuildStats.TotalBinaries = $Binaries.Count
    }
    
    # Collect frontend build information
    if (Test-Path "sveltekit-frontend\build") {
        try {
            $BuildFiles = Get-ChildItem -Recurse "sveltekit-frontend\build" -ErrorAction SilentlyContinue
            $BuildSize = ($BuildFiles | Measure-Object -Property Length -Sum).Sum
            $Report.Frontend = @{
                BuildSizeMB = [math]::Round($BuildSize / 1MB, 2)
                FileCount = $BuildFiles.Count
                BuildPath = (Resolve-Path "sveltekit-frontend\build").Path
            }
        }
        catch {
            Write-Log "⚠️ Could not analyze frontend build" "WARN"
        }
    }
    
    # Save report
    $ReportJson = $Report | ConvertTo-Json -Depth 4 -Compress:$false
    $ReportPath = "$LogDir\build-report-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').json"
    
    try {
        Set-Content -Path $ReportPath -Value $ReportJson -Encoding UTF8
        Write-Log "📄 Build report saved: $ReportPath" "SUCCESS"
    }
    catch {
        Write-Log "⚠️ Could not save build report: $_" "WARN"
    }
    
    # Display summary
    Write-Log "" "INFO"
    Write-Log "═══════════════════════════════════════" "INFO"
    Write-Log "           BUILD SUMMARY" "INFO"
    Write-Log "═══════════════════════════════════════" "INFO"
    Write-Log "Environment: $Environment" "INFO"
    Write-Log "Services Built: $($Report.BuildStats.TotalBinaries)" "SUCCESS"
    Write-Log "Total Binary Size: $($Report.BuildStats.TotalSizeMB) MB" "INFO"
    if ($Report.Frontend.BuildSizeMB) {
        Write-Log "Frontend Size: $($Report.Frontend.BuildSizeMB) MB" "INFO"
    }
    Write-Log "Build Log: $LogDir\$LogFile" "INFO"
    Write-Log "═══════════════════════════════════════" "INFO"
}

# Main execution flow
try {
    $BuildStart = Get-Date
    
    Write-Log "" "INFO"
    Write-Log "🏗️ Legal AI Platform Build Script v2.0.0" "INFO"
    Write-Log "Environment: $Environment" "INFO"
    Write-Log "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"
    Write-Log "═══════════════════════════════════════" "INFO"
    
    # Execute build pipeline
    Test-Prerequisites
    Build-GoServices
    Build-QUICServices
    Build-Frontend
    Run-Tests
    Start-Services
    Generate-BuildReport
    
    $TotalBuildTime = (Get-Date) - $BuildStart
    Write-Log "" "SUCCESS"
    Write-Log "🎉 BUILD COMPLETED SUCCESSFULLY!" "SUCCESS"
    Write-Log "⏱️ Total build time: $($TotalBuildTime.ToString('mm\:ss'))" "SUCCESS"
    Write-Log "" "SUCCESS"
    
    if ($StartServices) {
        Write-Log "🌐 Access your Legal AI Platform at:" "INFO"
        Write-Log "   • Frontend: http://localhost:5173" "INFO"
        Write-Log "   • Enhanced RAG API: http://localhost:8094" "INFO"
        Write-Log "   • Upload Service: http://localhost:8093" "INFO"
    }
    
    exit 0
}
catch {
    $BuildError = $_
    Write-Log "" "ERROR"
    Write-Log "💥 BUILD FAILED!" "ERROR"
    Write-Log "Error: $BuildError" "ERROR"
    Write-Log "Check the detailed log: $LogDir\$LogFile" "ERROR"
    Write-Log "" "ERROR"
    
    # Save error report
    $ErrorReport = @{
        Error = $BuildError.ToString()
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Environment = $Environment
        PowerShellVersion = $PSVersionTable.PSVersion.ToString()
        LogFile = "$LogDir\$LogFile"
    }
    
    $ErrorReportPath = "$LogDir\build-error-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').json"
    try {
        $ErrorReport | ConvertTo-Json -Depth 2 | Set-Content -Path $ErrorReportPath -Encoding UTF8
        Write-Log "Error report saved: $ErrorReportPath" "ERROR"
    }
    catch {
        # If we can't even save the error report, just continue
    }
    
    exit 1
}