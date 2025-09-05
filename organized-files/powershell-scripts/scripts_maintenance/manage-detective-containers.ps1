# Detective Evidence Synthesizer - Container Management Script
# Comprehensive container lifecycle management

param(
 [Parameter(Mandatory=$true)]
 [ValidateSet('create', 'start', 'stop', 'restart', 'destroy', 'status', 'logs', 'shell', 'backup', 'restore')]
 [string]$Action,
 
 [Parameter(Mandatory=$false)]
 [ValidateSet('cpu', 'gpu', 'both')]
 [string]$Version = 'cpu',
 
 [Parameter(Mandatory=$false)]
 [switch]$Force,
 
 [Parameter(Mandatory=$false)]
 [string]$Service,
 
 [Parameter(Mandatory=$false)]
 [string]$BackupPath = './backups'
)

# Configuration
$rootPath = 'C:\Users\james\Desktop\deeds-web\deeds-web-app'
$frontendPath = '$rootPath\sveltekit-frontend'

# Service definitions
$services = @{
 cpu = @{
 dockerFile = 'docker-compose.yml'
 envFile = '.env'
 workingDir = $frontendPath
 containers = @('detective_postgres_new', 'detective_qdrant_new', 'detective_redis_new', 'detective_rabbitmq_new', 'detective_ollama_new')
 ports = @('5433', '6334', '6380', '5673', '15673', '11435')
 }
 gpu = @{
 dockerFile = 'docker-compose.gpu.yml'
 envFile = '.env.gpu'
 workingDir = $rootPath
 containers = @('detective_postgres_gpu', 'detective_qdrant_gpu', 'detective_redis_gpu', 'detective_rabbitmq_gpu', 'detective_ollama_gpu')
 ports = @('5434', '6335', '6381', '5674', '15674', '11436')
 }
}

function Write-Header {
 param([string]$Title)
 Write-Host "
🕵️ Detective Evidence Synthesizer - $Title" -ForegroundColor Cyan
 Write-Host ('=' * 60) -ForegroundColor Gray
}

function Write-Success {
 param([string]$Message)
 Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
 param([string]$Message)
 Write-Host "⚠️ $Message" -ForegroundColor Yellow
}

function Write-Error {
 param([string]$Message)
 Write-Host "❌ $Message" -ForegroundColor Red
}

function Test-DockerRunning {
 try {
 docker info | Out-Null
 return $true
 } catch {
 return $false
 }
}

function Get-ContainerStatus {
 param([string]$ContainerName)
 $status = docker ps -a --filter "name=$ContainerName" --format "{{.Status}}" 2>$null
 if ($status) {
 if ($status -like '*Up*') {
 return 'Running'
 } elseif ($status -like '*Exited*') {
 return 'Stopped'
 } else {
 return 'Unknown'
 }
 }
 return 'Not Found'
}

function Start-DetectiveServices {
 param([string]$Version)
 
 $config = $services[$Version]
 Write-Header "Starting $($Version.ToUpper()) Services"
 
 Set-Location $config.workingDir
 
 if ($Version -eq 'gpu') {
 Copy-Item "$rootPath\.env.gpu" "$frontendPath\.env" -Force
 Write-Success "GPU environment configuration applied"
 }
 
 try {
 if ($Version -eq 'gpu') {
 docker-compose -f $config.dockerFile up -d
 } else {
 docker-compose up -d
 }
 Write-Success "$($Version.ToUpper()) services started successfully"
 
 # Wait for services to initialize
 Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
 Start-Sleep -Seconds 10
 
 # Setup local model
 Set-Location $rootPath
 if ($Version -eq 'gpu') {
 & './setup-local-model-gpu.ps1'
 } else {
 & './setup-local-model.ps1'
 }
 
 } catch {
 Write-Error "Failed to start services: $_"
 return $false
 }
 
 return $true
}

function Stop-DetectiveServices {
 param([string]$Version)
 
 $config = $services[$Version]
 Write-Header "Stopping $($Version.ToUpper()) Services"
 
 Set-Location $config.workingDir
 
 try {
 if ($Version -eq 'gpu') {
 docker-compose -f $config.dockerFile down
 } else {
 docker-compose down
 }
 Write-Success "$($Version.ToUpper()) services stopped successfully"
 } catch {
 Write-Error "Failed to stop services: $_"
 return $false
 }
 
 return $true
}

function Show-ServiceStatus {
 param([string]$Version)
 
 $config = $services[$Version]
 Write-Header "$($Version.ToUpper()) Service Status"
 
 foreach ($container in $config.containers) {
 $status = Get-ContainerStatus $container
 $color = switch ($status) {
 'Running' { 'Green' }
 'Stopped' { 'Yellow' }
 'Not Found' { 'Red' }
 default { 'Gray' }
 }
 Write-Host " $container : $status" -ForegroundColor $color
 }
 
 Write-Host "
Service Endpoints:" -ForegroundColor Cyan
 if ($Version -eq 'cpu') {
 Write-Host " • PostgreSQL: localhost:5433" -ForegroundColor White
 Write-Host " • Qdrant: localhost:6334 (Dashboard: http://localhost:6334/dashboard)" -ForegroundColor White
 Write-Host " • Redis: localhost:6380" -ForegroundColor White
 Write-Host " • RabbitMQ: localhost:5673 (Management: http://localhost:15673)" -ForegroundColor White
 Write-Host " • Ollama: localhost:11435 (Model: gemma3-detective)" -ForegroundColor White
 } else {
 Write-Host " • PostgreSQL: localhost:5434" -ForegroundColor White
 Write-Host " • Qdrant: localhost:6335 (Dashboard: http://localhost:6335/dashboard)" -ForegroundColor White
 Write-Host " • Redis: localhost:6381" -ForegroundColor White
 Write-Host " • RabbitMQ: localhost:5674 (Management: http://localhost:15674)" -ForegroundColor White
 Write-Host " • Ollama GPU: localhost:11436 (Model: gemma3-detective)" -ForegroundColor White
 }
}

function Show-ServiceLogs {
 param([string]$Version, [string]$Service)
 
 $config = $services[$Version]
 Set-Location $config.workingDir
 
 if ($Service) {
 Write-Header "Logs for $Service ($($Version.ToUpper()))"
 if ($Version -eq 'gpu') {
 docker-compose -f $config.dockerFile logs -f $Service
 } else {
 docker-compose logs -f $Service
 }
 } else {
 Write-Header "All Service Logs ($($Version.ToUpper()))"
 if ($Version -eq 'gpu') {
 docker-compose -f $config.dockerFile logs -f
 } else {
 docker-compose logs -f
 }
 }
}

function Open-ServiceShell {
 param([string]$Version, [string]$Service)
 
 $config = $services[$Version]
 
 if (-not $Service) {
 Write-Warning "Please specify a service with -Service parameter"
 Write-Host "Available services: $($config.containers -join ', ')" -ForegroundColor White
 return
 }
 
 $containerName = $config.containers | Where-Object { $_ -like "*$Service*" } | Select-Object -First 1
 
 if (-not $containerName) {
 Write-Error "Service '$Service' not found"
 return
 }
 
 Write-Header "Opening shell for $containerName"
 docker exec -it $containerName /bin/bash
}

# Main script logic
Set-Location $rootPath

# Check Docker
if (-not (Test-DockerRunning)) {
 Write-Error "Docker is not running. Please start Docker Desktop."
 exit 1
}

# Load environment variables
if (Test-Path '.env.docker') {
 Get-Content '.env.docker' | ForEach-Object {
 if ($_ -match '^([^#][^=]+)=(.*)$') {
 [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
 }
 }
}

# Execute action
switch ($Action) {
 'create' {
 if ($Version -eq 'both') {
 & './create-detective-containers.ps1'
 & './create-detective-containers.ps1' -GPU
 } elseif ($Version -eq 'gpu') {
 & './create-detective-containers.ps1' -GPU
 } else {
 & './create-detective-containers.ps1'
 }
 }
 
 'start' {
 if ($Version -eq 'both') {
 Start-DetectiveServices 'cpu'
 Start-DetectiveServices 'gpu'
 } else {
 Start-DetectiveServices $Version
 }
 }
 
 'stop' {
 if ($Version -eq 'both') {
 Stop-DetectiveServices 'cpu'
 Stop-DetectiveServices 'gpu'
 } else {
 Stop-DetectiveServices $Version
 }
 }
 
 'restart' {
 if ($Version -eq 'both') {
 Stop-DetectiveServices 'cpu'
 Stop-DetectiveServices 'gpu'
 Start-Sleep -Seconds 5
 Start-DetectiveServices 'cpu'
 Start-DetectiveServices 'gpu'
 } else {
 Stop-DetectiveServices $Version
 Start-Sleep -Seconds 5
 Start-DetectiveServices $Version
 }
 }
 
 'destroy' {
 if ($Force -or (Read-Host "Are you sure you want to destroy all containers and data? (y/N)") -eq 'y') {
 Write-Header "Destroying All Detective Containers"
 
 # Stop and remove containers
 $allContainers = $services.cpu.containers + $services.gpu.containers
 foreach ($container in $allContainers) {
 docker stop $container 2>$null | Out-Null
 docker rm $container 2>$null | Out-Null
 }
 
 # Remove volumes
 docker volume rm detective-evidence-synthesizer_postgres_data 2>$null | Out-Null
 docker volume rm detective-evidence-synthesizer_qdrant_data 2>$null | Out-Null
 docker volume rm detective-evidence-synthesizer_redis_data 2>$null | Out-Null
 docker volume rm detective-evidence-synthesizer_rabbitmq_data 2>$null | Out-Null
 docker volume rm detective-evidence-synthesizer_ollama_data 2>$null | Out-Null
 
 # Remove networks
 docker network rm detective_network_new 2>$null | Out-Null
 docker network rm detective_network_gpu 2>$null | Out-Null
 
 Write-Success "All Detective containers and data destroyed"
 } else {
 Write-Warning "Operation cancelled"
 }
 }
 
 'status' {
 if ($Version -eq 'both') {
 Show-ServiceStatus 'cpu'
 Show-ServiceStatus 'gpu'
 } else {
 Show-ServiceStatus $Version
 }
 }
 
 'logs' {
 Show-ServiceLogs $Version $Service
 }
 
 'shell' {
 Open-ServiceShell $Version $Service
 }
 
 'backup' {
 Write-Header "Backing up Detective Data"
 # Implementation for backup functionality
 Write-Warning "Backup functionality coming soon"
 }
 
 'restore' {
 Write-Header "Restoring Detective Data"
 # Implementation for restore functionality
 Write-Warning "Restore functionality coming soon"
 }
}

Write-Host "
🎉 Operation completed!" -ForegroundColor Green