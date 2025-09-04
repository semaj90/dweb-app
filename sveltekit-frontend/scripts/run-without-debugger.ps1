# PowerShell script to run npm commands without VS Code debugger
param(
    [Parameter(Mandatory=$true)]
    [string]$Command,
    [Parameter(Mandatory=$false)]
    [string[]]$AdditionalArgs
)

# Clear all Node.js debugger environment variables that cause debugger attachment
Remove-Item Env:NODE_OPTIONS -ErrorAction SilentlyContinue
Remove-Item Env:NODE_INSPECT -ErrorAction SilentlyContinue  
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
Remove-Item Env:VSCODE_NODE_CACHED_DATA_DIR -ErrorAction SilentlyContinue
Remove-Item Env:ELECTRON_ENABLE_LOGGING -ErrorAction SilentlyContinue

# Set clean Node options
$env:NODE_OPTIONS = "--max-old-space-size=8192"
$env:NODE_INSPECT = ""

# Build full command with additional arguments
$fullCommand = $Command
if ($AdditionalArgs -and $AdditionalArgs.Count -gt 0) {
    $fullCommand += " " + ($AdditionalArgs -join " ")
}

# Execute the command
Write-Host "Running: $fullCommand" -ForegroundColor Green
Invoke-Expression $fullCommand