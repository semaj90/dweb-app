# GPU-Accelerated Rapid JSON Parser Setup Script (PowerShell Wrapper)
# Runs the TypeScript setup script with proper Node.js execution

param(
    [switch]$NoWasm,
    [switch]$NoTests,
    [switch]$NoDocker,
    [switch]$NoVscode,
    [switch]$NoGpu,
    [switch]$Verbose,
    [switch]$Help
)

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Cyan"
    White = "White"
}

function Write-StatusInfo {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

if ($Help) {
    Write-Host @"
GPU-Accelerated Rapid JSON Parser Setup

Usage: .\setup-gpu-json-parser.ps1 [options]

Options:
  -NoWasm       Skip WebAssembly build
  -NoTests      Skip running tests
  -NoDocker     Skip Docker optimization
  -NoVscode     Skip VS Code configuration
  -NoGpu        Disable GPU acceleration
  -Verbose      Enable verbose output
  -Help         Show this help message

Examples:
  .\setup-gpu-json-parser.ps1                    # Full setup
  .\setup-gpu-json-parser.ps1 -NoWasm -NoTests   # Config only
  .\setup-gpu-json-parser.ps1 -Verbose           # Full setup with verbose output
"@ -ForegroundColor $Colors.White
    exit 0
}

Write-StatusInfo "🚀 Starting GPU-Accelerated JSON Parser setup..."

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    Write-Success "Node.js found: $nodeVersion"
} catch {
    Write-ErrorMsg "Node.js not found. Please install Node.js 18+ and try again."
    exit 1
}

# Check TypeScript support
try {
    $tsVersion = npx tsc --version 2>$null
    Write-Success "TypeScript found: $tsVersion"
} catch {
    Write-Warning "TypeScript not found globally. Installing..."
    npm install -g typescript
}

# Build command line arguments
$args = @()

if ($NoWasm) { $args += "--no-wasm" }
if ($NoTests) { $args += "--no-tests" }
if ($NoDocker) { $args += "--no-docker" }
if ($NoVscode) { $args += "--no-vscode" }
if ($NoGpu) { $args += "--no-gpu" }
if ($Verbose) { $args += "--verbose" }

Write-StatusInfo "Running setup with arguments: $($args -join ' ')"

# Execute the TypeScript setup script
try {
    if ($args.Count -gt 0) {
        npx tsx setup-gpu-json-parser.ts @args
    } else {
        npx tsx setup-gpu-json-parser.ts
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Success "🎉 Setup completed successfully!"
        Write-StatusInfo ""
        Write-StatusInfo "Next steps:"
        Write-StatusInfo "1. Review the generated INTEGRATION_GUIDE.md"
        Write-StatusInfo "2. Test the WebAssembly module: npm run test:wasm"
        Write-StatusInfo "3. Run benchmarks: npm run benchmark:json"
        Write-StatusInfo "4. Start development: npm run dev"
        Write-StatusInfo ""
        Write-StatusInfo "VS Code Commands available:"
        Write-StatusInfo "- Ctrl+Shift+P > 'GPU JSON Parser: Parse'"
        Write-StatusInfo "- Ctrl+Shift+P > 'GPU JSON Parser: Benchmark'"
        Write-StatusInfo "- Ctrl+Shift+P > 'GPU JSON Parser: Metrics'"
    } else {
        Write-ErrorMsg "Setup failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-ErrorMsg "Setup failed: $($_.Exception.Message)"
    Write-StatusInfo ""
    Write-StatusInfo "Troubleshooting:"
    Write-StatusInfo "1. Ensure Node.js 18+ is installed"
    Write-StatusInfo "2. Run 'npm install' to install dependencies"
    Write-StatusInfo "3. Check that TypeScript is available"
    Write-StatusInfo "4. Try running with -Verbose for more details"
    exit 1
}
