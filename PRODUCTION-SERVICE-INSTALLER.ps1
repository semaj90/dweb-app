# PRODUCTION-SERVICE-INSTALLER.ps1
# Install as Windows Service

param([switch]$Uninstall)

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Run as Administrator"
    exit 1
}

$serviceName = "SIMD-Redis-Server"
$binaryPath = "$PWD\go-microservice\simd-server-prod.exe"

if ($Uninstall) {
    sc.exe delete $serviceName
    Write-Host "Service removed"
} else {
    # Create service
    New-Service -Name $serviceName `
                -BinaryPathName $binaryPath `
                -DisplayName "SIMD Redis Server" `
                -StartupType Automatic `
                -Description "High-performance SIMD JSON parser with Redis"
    
    # Configure recovery
    sc.exe failure $serviceName reset=86400 actions=restart/5000/restart/10000/restart/30000
    
    Start-Service $serviceName
    Write-Host "Service installed and started"
}
