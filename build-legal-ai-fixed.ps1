# Legal AI Platform - Production Build Automation Script
# Version: 2.0.0 - Fixed
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
    try {
        Add-Content -Path "$LogDir\$LogFile" -Value $LogMessage
    } catch {
        # Ignore logging errors
    }
}

function Test-Prerequisites {
    Write-Log "Checking build prerequisites..." "INFO"
    
    # Test Go
    try {
        $goOutput = go version 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Go not found" }
        Write-Log "Go: $goOutput" "SUCCESS"
    } catch {
        Write-Log "Go check failed: $_" "ERROR"
        throw "Go is required but not found"
    }
    
    # Test Node.js
    try {
        $nodeOutput = node --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Node.js not found" }
        Write-Log "Node.js: $nodeOutput" "SUCCESS"
    } catch {
        Write-Log "Node.js check failed: $_" "ERROR"
        throw "Node.js is required but not found"
    }
    
    # Test NPM
    try {
        $npmOutput = npm --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw "NPM not found" }
        Write-Log "NPM: $npmOutput" "SUCCESS"
    } catch {
        Write-Log "NPM check failed: $_" "ERROR"
        throw "NPM is required but not found"
    }
    
    # Check required directories
    $RequiredDirs = @("go-microservice", "quic-services", "sveltekit-frontend")
    foreach ($dir in $RequiredDirs) {
        if (!(Test-Path $dir)) {
            throw "Required directory not found: $dir"
        }
    }
    
    Write-Log "All prerequisites satisfied" "SUCCESS"
}

function Build-GoServices {
    Write-Log "Building Go microservices..." "INFO"
    
    $Services = @(
        "enhanced-rag",
        "upload-service", 
        "grpc-server"
    )
    
    Push-Location go-microservice
    
    try {
        # Clean build if requested
        if ($CleanBuild) {
            Write-Log "Cleaning Go module cache and binaries..." "INFO"
            go clean -modcache 2>$null
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
        Write-Log "Updating Go dependencies..." "INFO"
        go mod tidy
        if ($LASTEXITCODE -ne 0) { 
            Write-Log "go mod tidy failed" "ERROR"
            throw "Go mod tidy failed" 
        }
        
        # Build services
        $BuiltServices = @()
        foreach ($service in $Services) {
            $ServicePath = ".\cmd\$service"
            if (!(Test-Path $ServicePath)) {
                Write-Log "Service path not found: $ServicePath, skipping $service" "WARN"
                continue
            }
            
            Write-Log "Building $service..." "INFO"
            
            $BuildStart = Get-Date
            go build -o ".\bin\$service.exe" ".\cmd\$service"
            
            if ($LASTEXITCODE -ne 0) {
                Write-Log "Build failed for $service" "ERROR"
                throw "Build failed for $service"
            }
            
            $BuildTime = (Get-Date) - $BuildStart
            
            if (Test-Path ".\bin\$service.exe") {
                $FileInfo = Get-Item ".\bin\$service.exe"
                $SizeMB = [math]::Round($FileInfo.Length / 1MB, 2)
                $BuildSeconds = $BuildTime.TotalSeconds.ToString('F2')
                
                Write-Log "$service built successfully (${SizeMB}MB, ${BuildSeconds}s)" "SUCCESS"
                $BuiltServices += $service
            } else {
                throw "Binary not created for $service"
            }
        }
        
        Write-Log "Built $($BuiltServices.Count) Go services successfully" "SUCCESS"
        return $BuiltServices
        
    } finally {
        Pop-Location
    }
}

function Build-QUICServices {
    Write-Log "Building QUIC protocol services..." "INFO"
    
    if (!(Test-Path "quic-services")) {
        Write-Log "QUIC services directory not found, skipping" "WARN"
        return @()
    }
    
    Push-Location quic-services
    
    try {
        # Update dependencies
        go mod tidy
        if ($LASTEXITCODE -ne 0) {
            Write-Log "QUIC services go mod tidy failed" "WARN"
            return @()
        }
        
        $QUICServices = @(
            @{Name="quic-gateway"; Source="quic-gateway.go"},
            @{Name="quic-vector-proxy"; Source="quic-vector-proxy.go"},
            @{Name="quic-ai-stream"; Source="quic-ai-stream.go"}
        )
        
        $BuiltQUICServices = @()
        foreach ($service in $QUICServices) {
            if (Test-Path $service.Source) {
                Write-Log "Building $($service.Name)..." "INFO"
                
                $BuildStart = Get-Date
                go build -o "..\go-microservice\bin\$($service.Name).exe" $service.Source
                
                if ($LASTEXITCODE -eq 0) {
                    $BuildTime = (Get-Date) - $BuildStart
                    $BuildSeconds = $BuildTime.TotalSeconds.ToString('F2')
                    Write-Log "$($service.Name) built successfully (${BuildSeconds}s)" "SUCCESS"
                    $BuiltQUICServices += $service.Name
                } else {
                    Write-Log "Build failed for $($service.Name)" "WARN"
                }
            } else {
                Write-Log "Source not found for $($service.Name): $($service.Source)" "WARN"
            }
        }
        
        if ($BuiltQUICServices.Count -gt 0) {
            Write-Log "Built $($BuiltQUICServices.Count) QUIC services" "SUCCESS"
        }
        
        return $BuiltQUICServices
        
    } finally {
        Pop-Location
    }
}

function Build-Frontend {
    Write-Log "Building SvelteKit frontend..." "INFO"
    
    if (!(Test-Path "sveltekit-frontend")) {
        Write-Log "SvelteKit frontend directory not found, skipping" "WARN"
        return
    }
    
    Push-Location sveltekit-frontend
    
    try {
        # Check if package.json exists
        if (!(Test-Path "package.json")) {
            Write-Log "package.json not found in sveltekit-frontend directory" "ERROR"
            throw "package.json not found"
        }
        
        # Install dependencies
        Write-Log "Installing npm dependencies..." "INFO"
        $InstallStart = Get-Date
        
        if ($CleanBuild -and (Test-Path "node_modules")) {
            Write-Log "Cleaning node_modules..." "INFO"
            Remove-Item -Recurse -Force node_modules
        }
        
        npm ci --silent 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "npm ci failed, trying npm install..." "WARN"
            npm install --silent 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Log "npm install failed" "ERROR"
                throw "npm install failed"
            }
        }
        
        $InstallTime = (Get-Date) - $InstallStart
        $InstallSeconds = $InstallTime.TotalSeconds.ToString('F2')
        Write-Log "Dependencies installed (${InstallSeconds}s)" "SUCCESS"
        
        # TypeScript check
        Write-Log "Running TypeScript checks..." "INFO"
        npm run check 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "TypeScript check had issues, continuing..." "WARN"
        } else {
            Write-Log "TypeScript checks passed" "SUCCESS"
        }
        
        # Build application
        $BuildStart = Get-Date
        Write-Log "Building frontend..." "INFO"
        npm run build 2>$null
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Frontend build failed" "ERROR"
            throw "Frontend build failed"
        }
        
        $BuildTime = (Get-Date) - $BuildStart
        $BuildSeconds = $BuildTime.TotalSeconds.ToString('F2')
        Write-Log "Frontend built successfully (${BuildSeconds}s)" "SUCCESS"
        
        # Check build output
        if (Test-Path "build") {
            try {
                $BuildFiles = Get-ChildItem -Recurse build
                $BuildSize = ($BuildFiles | Measure-Object -Property Length -Sum).Sum
                $BuildSizeMB = [math]::Round($BuildSize / 1MB, 2)
                Write-Log "Build output: ${BuildSizeMB}MB" "INFO"
            } catch {
                Write-Log "Could not calculate build size" "WARN"
            }
        }
        
    } finally {
        Pop-Location
    }
}

function Run-Tests {
    if ($SkipTests) {
        Write-Log "Skipping tests as requested" "WARN"
        return
    }
    
    Write-Log "Running test suite..." "INFO"
    
    # Go tests
    if (Test-Path "go-microservice") {
        Push-Location go-microservice
        try {
            Write-Log "Running Go unit tests..." "INFO"
            $TestStart = Get-Date
            
            go test -v ./... 2>$null
            if ($LASTEXITCODE -eq 0) {
                $TestTime = (Get-Date) - $TestStart
                $TestSeconds = $TestTime.TotalSeconds.ToString('F2')
                Write-Log "Go tests completed (${TestSeconds}s)" "SUCCESS"
            } else {
                Write-Log "Some Go tests failed or not configured" "WARN"
            }
        } finally {
            Pop-Location
        }
    }
    
    # Frontend tests (if they exist)
    if (Test-Path "sveltekit-frontend") {
        Push-Location sveltekit-frontend
        try {
            if (Test-Path "vitest.config.ts") {
                Write-Log "Running frontend tests..." "INFO"
                npm run test:unit 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "Frontend tests passed" "SUCCESS"
                } else {
                    Write-Log "Frontend tests failed or not configured" "WARN"
                }
            } else {
                Write-Log "No frontend tests configured" "INFO"
            }
        } finally {
            Pop-Location
        }
    }
}

function Generate-BuildReport {
    Write-Log "Generating build report..." "INFO"
    
    # Get Git information
    $GitVersion = ""
    $GitCommit = ""
    $GitBranch = ""
    
    try {
        $GitVersion = git describe --tags --always 2>$null
        $GitCommit = git rev-parse HEAD 2>$null  
        $GitBranch = git rev-parse --abbrev-ref HEAD 2>$null
    } catch {
        Write-Log "Could not retrieve Git information" "WARN"
    }
    
    $Report = @{
        BuildInfo = @{
            BuildTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Environment = $Environment
            Version = $GitVersion
            Commit = $GitCommit
            Branch = $GitBranch
        }
        Services = @()
        BuildStats = @{
            TotalBinaries = 0
            TotalSizeMB = 0
        }
    }
    
    # Collect service binary information
    if (Test-Path "go-microservice\bin") {
        try {
            $Binaries = Get-ChildItem "go-microservice\bin\*.exe"
            foreach ($binary in $Binaries) {
                $SizeMB = [math]::Round($binary.Length / 1MB, 2)
                $Report.Services += @{
                    Name = $binary.BaseName
                    Size = "$SizeMB MB"
                    Modified = $binary.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
                }
                $Report.BuildStats.TotalSizeMB += $SizeMB
            }
            $Report.BuildStats.TotalBinaries = $Binaries.Count
        } catch {
            Write-Log "Could not analyze binaries" "WARN"
        }
    }
    
    # Save report
    try {
        $ReportJson = $Report | ConvertTo-Json -Depth 3
        $ReportPath = "$LogDir\build-report-$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss').json"
        Set-Content -Path $ReportPath -Value $ReportJson -Encoding UTF8
        Write-Log "Build report saved: $ReportPath" "SUCCESS"
    } catch {
        Write-Log "Could not save build report: $_" "WARN"
    }
    
    # Display summary
    Write-Log "" "INFO"
    Write-Log "==============================================" "INFO"
    Write-Log "           BUILD SUMMARY" "INFO"
    Write-Log "==============================================" "INFO"
    Write-Log "Environment: $Environment" "INFO"
    Write-Log "Services Built: $($Report.BuildStats.TotalBinaries)" "SUCCESS"
    Write-Log "Total Binary Size: $($Report.BuildStats.TotalSizeMB) MB" "INFO"
    Write-Log "Build Log: $LogDir\$LogFile" "INFO"
    Write-Log "==============================================" "INFO"
}

# Main execution flow
try {
    $BuildStart = Get-Date
    
    Write-Log "" "INFO"
    Write-Log "Legal AI Platform Build Script v2.0.0" "INFO"
    Write-Log "Environment: $Environment" "INFO"
    Write-Log "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "INFO"
    Write-Log "==============================================" "INFO"
    
    # Execute build pipeline
    Test-Prerequisites
    $GoServices = Build-GoServices
    $QUICServices = Build-QUICServices
    Build-Frontend
    Run-Tests
    Generate-BuildReport
    
    $TotalBuildTime = (Get-Date) - $BuildStart
    $TotalSeconds = $TotalBuildTime.TotalSeconds.ToString('F2')
    
    Write-Log "" "SUCCESS"
    Write-Log "BUILD COMPLETED SUCCESSFULLY!" "SUCCESS"
    Write-Log "Total build time: ${TotalSeconds}s" "SUCCESS"
    Write-Log "" "SUCCESS"
    
    Write-Log "Access your Legal AI Platform at:" "INFO"
    Write-Log "  Frontend: http://localhost:5173" "INFO"
    Write-Log "  Enhanced RAG API: http://localhost:8094" "INFO"
    Write-Log "  Upload Service: http://localhost:8093" "INFO"
    
    exit 0
}
catch {
    $BuildError = $_
    Write-Log "" "ERROR"
    Write-Log "BUILD FAILED!" "ERROR"
    Write-Log "Error: $BuildError" "ERROR"
    Write-Log "Check the detailed log: $LogDir\$LogFile" "ERROR"
    Write-Log "" "ERROR"
    
    exit 1
}