#!/usr/bin/env powershell
# Phase 4: Complete Data Management & Event Streaming Setup
# Configures Neo4j, Service Workers, and integration testing

param(
    [switch]$SkipContainers,
    [switch]$TestOnly,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "Phase 4: Data Management & Event Streaming Setup" -ForegroundColor Cyan
Write-Host "--------------------------------------------------------------" -ForegroundColor Cyan

# Configuration
$PHASE4_SERVICES = @(
    "deeds-neo4j",
    "deeds-postgres",
    "deeds-redis",
    "deeds-rabbitmq",
    "deeds-qdrant",
    "deeds-ollama-gpu"
)

function Write-Progress {
    param($Message, $Color = "Green")
    Write-Host "OK: $Message" -ForegroundColor $Color
}

function Write-Error {
    param($Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Write-Info {
    param($Message)
    Write-Host "INFO:  $Message" -ForegroundColor Blue
}

function Test-DockerService {
    param($ServiceName)
    try {
        $status = docker ps --filter "name=$ServiceName" --format "{{.Status}}" 2>$null
        return $status -like "*Up*"
    }
    catch {
        return $false
    }
}

function Test-Port {
    param($Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.ConnectAsync("localhost", $Port).Wait(2000)
        $result = $connection.Connected
        $connection.Close()
        return $result
    }
    catch {
        return $false
    }
}

# Check Prerequisites
Write-Info "Checking prerequisites..."

# Check Docker
try {
    docker --version | Out-Null
    Write-Progress "Docker is available"
}
catch {
    Write-Error "Docker is not installed or not running"
    exit 1
}

# Check Node.js
try {
    node --version | Out-Null
    Write-Progress "Node.js is available"
}
catch {
    Write-Error "Node.js is not installed"
    exit 1
}

# Step 1: Install Dependencies
if (-not $TestOnly) {
    Write-Info "Installing Phase 4 dependencies..."

    # Check if package.json exists
    if (-not (Test-Path "package.json")) {
        Write-Error "package.json not found. Run from project root directory."
        exit 1
    }

    try {
        # Install required dependencies
        npm install neo4j-driver@5.15.0 ws@8.18.0 amqplib@0.10.3 ioredis@5.6.1 --save
        Write-Progress "Dependencies installed successfully"
    }
    catch {
        Write-Error "Failed to install dependencies: $_"
        exit 1
    }
}

# Step 2: Start Docker Services
if (-not $SkipContainers -and -not $TestOnly) {
    Write-Info "Starting Phase 4 Docker services..."

    # Check if docker-compose.yml exists
    $dockerComposeFile = ""
    if (Test-Path "docker-compose-optimized.yml") {
        $dockerComposeFile = "docker-compose-optimized.yml"
    }
    elseif (Test-Path "docker-compose.yml") {
        $dockerComposeFile = "docker-compose.yml"
    }
    else {
        Write-Error "No docker-compose file found"
        exit 1
    }

    try {
        # Start services with proper compose file
        docker-compose -f $dockerComposeFile up -d --remove-orphans

        Write-Info "Waiting for services to be ready..."
        Start-Sleep -Seconds 30

        # Check service health with fallback names
        $actualServices = @{}
        foreach ($service in $PHASE4_SERVICES) {
            $containerExists = docker ps -a --filter "name=$service" --format "{{.Names}}" 2>$null
            if (-not $containerExists) {
                # Try alternative naming patterns
                $altName = $service -replace "deeds-", "legal-"
                $containerExists = docker ps -a --filter "name=$altName" --format "{{.Names}}" 2>$null
                if ($containerExists) {
                    $actualServices[$service] = $altName
                }
            }
            else {
                $actualServices[$service] = $service
            }
        }

        foreach ($service in $actualServices.Keys) {
            $actualName = $actualServices[$service]
            if (Test-DockerService $actualName) {
                Write-Progress "$service ($actualName) is running"
            }
            else {
                Write-Error "$service ($actualName) failed to start"
            }
        }
    }
    catch {
        Write-Error "Failed to start Docker services: $_"
        Write-Info "Trying to start individual services..."

        # Fallback: start essential services individually
        try {
            docker run -d --name temp-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/LegalSecure2024! neo4j:5.15-community
            Write-Progress "Started Neo4j container manually"
        }
        catch {
            Write-Error "Failed to start Neo4j manually"
        }
    }
}

# Step 3: Test Service Connectivity
Write-Info "Testing service connectivity..."

$serviceTests = @{
    "Neo4j HTTP" = @{ Port = 7474; Required = $true }
    "Neo4j Bolt" = @{ Port = 7687; Required = $true }
    "PostgreSQL" = @{ Port = 5432; Required = $false }
    "Redis" = @{ Port = 6379; Required = $false }
    "RabbitMQ" = @{ Port = 5672; Required = $false }
    "RabbitMQ Management" = @{ Port = 15672; Required = $false }
    "Qdrant" = @{ Port = 6333; Required = $false }
    "Ollama" = @{ Port = 11434; Required = $false }
}

foreach ($serviceName in $serviceTests.Keys) {
    $port = $serviceTests[$serviceName].Port
    $required = $serviceTests[$serviceName].Required

    if (Test-Port $port) {
        Write-Progress "$serviceName (port $port) is accessible"
    }
    else {
        if ($required) {
            Write-Error "$serviceName (port $port) is not accessible - REQUIRED SERVICE"
        }
        else {
            Write-Host "WARN:  $serviceName (port $port) is not accessible - optional" -ForegroundColor Yellow
        }
    }
}

# Step 4: Initialize Neo4j Schema (if accessible)
if (Test-Port 7474) {
    Write-Info "Initializing Neo4j graph schema..."

    try {
        # Find the actual Neo4j container
        $neo4jContainer = docker ps --filter "publish=7474" --format "{{.Names}}" | Select-Object -First 1

        if ($neo4jContainer) {
            Write-Info "Found Neo4j container: $neo4jContainer"

            # Wait for Neo4j to be fully ready
            $retries = 10
            $ready = $false

            while ($retries -gt 0 -and -not $ready) {
                try {
                    $testResult = docker exec $neo4jContainer cypher-shell -u neo4j -p LegalSecure2024! "RETURN 1 as test;" 2>$null
                    if ($testResult -like '*test*') {
                        $ready = $true
                        Write-Progress "Neo4j is ready for queries"
                    }
                }
                catch {
                    Start-Sleep -Seconds 3
                    $retries--
                }
            }

            if ($ready) {
                # Create basic schema
                $schemaCommands = @(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",
                    "CREATE INDEX IF NOT EXISTS FOR (c:Case) ON (c.status);",
                    "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name);"
                )

                foreach ($command in $schemaCommands) {
                    try {
                        docker exec $neo4jContainer cypher-shell -u neo4j -p LegalSecure2024! -- "$command" 2>$null
                    }
                    catch {
                        Write-Host "WARN:  Schema command failed (may already exist): $command" -ForegroundColor Yellow
                    }
                }

                Write-Progress "Neo4j schema initialized"
            }
            else {
                Write-Error "Neo4j not ready after retries"
            }
        }
        else {
            Write-Error "No Neo4j container found on port 7474"
        }
    }
    catch {
        Write-Error "Neo4j initialization failed: $_"
    }
}

# Step 5: Create Service Integration Files
Write-Info "Creating service integration files..."

# Create basic service connection test
$serviceTestContent = @'
// Phase 4 Service Connection Test
import { createConnection } from 'net';

const services = [
    { name: 'Neo4j HTTP', port: 7474 },
    { name: 'Neo4j Bolt', port: 7687 },
    { name: 'PostgreSQL', port: 5432 },
    { name: 'Redis', port: 6379 },
    { name: 'RabbitMQ', port: 5672 },
    { name: 'Qdrant', port: 6333 },
    { name: 'Ollama', port: 11434 }
];

async function testService(service) {
    return new Promise((resolve) => {
        const socket = createConnection(service.port, 'localhost');

        socket.on('connect', () => {
            socket.destroy();
            resolve({ ...service, status: 'connected' });
        });

        socket.on('error', () => {
            resolve({ ...service, status: 'failed' });
        });

        setTimeout(() => {
            socket.destroy();
            resolve({ ...service, status: 'timeout' });
        }, 2000);
    });
}

console.log('Testing Phase 4 service connections...');

Promise.all(services.map(testService)).then(results => {
    results.forEach(result => {
        const status = result.status === 'connected' ? 'OK' : 'FAIL';
        console.log(`${status} ${result.name} (port ${result.port}): ${result.status}`);
    });

    const connectedCount = results.filter(r => r.status === 'connected').length;
    console.log(`\n ${connectedCount}/${results.length} services connected`);

    process.exit(connectedCount > 0 ? 0 : 1);
});
'@

try {
    $serviceTestContent | Out-File -FilePath "test-phase4-services.mjs" -Encoding UTF8

    # Run service test
    $testOutput = node test-phase4-services.mjs 2>&1
    Write-Host $testOutput

    if ($LASTEXITCODE -eq 0) {
        Write-Progress "Service connection test passed"
    }
    else {
        Write-Error "Some services are not accessible"
    }

    # Cleanup
    Remove-Item "test-phase4-services.mjs" -ErrorAction SilentlyContinue
}
catch {
    Write-Error "Service testing failed: $_"
}

# Step 6: Health Check Summary
Write-Info "Performing final health check..."

$healthResults = @{}

# Check essential services
if (Test-Port 7474) {
    $healthResults["Neo4j"] = "OK Connected"
}
else {
    $healthResults["Neo4j"] = "FAIL Not accessible"
}

# Check optional services
$optionalServices = @{
    "PostgreSQL" = 5432
    "Redis" = 6379
    "RabbitMQ" = 5672
    "Qdrant" = 6333
    "Ollama" = 11434
}

foreach ($serviceName in $optionalServices.Keys) {
    $port = $optionalServices[$serviceName]
    if (Test-Port $port) {
        $healthResults[$serviceName] = "OK Connected"
    }
    else {
        $healthResults[$serviceName] = "WARN  Not available"
    }
}

# Display results
Write-Host "`n Phase 4 Service Status:" -ForegroundColor Cyan
Write-Host "--------------------------------------------------------------" -ForegroundColor Cyan

foreach ($service in $healthResults.Keys) {
    Write-Host "$service : $($healthResults[$service])"
}

# Check if essential services are running
$neo4jRunning = $healthResults["Neo4j"] -like "*OK*"

if ($neo4jRunning) {
    Write-Host "`n Phase 4: Data Management and Event Streaming Setup Complete!" -ForegroundColor Green
    Write-Host "--------------------------------------------------------------" -ForegroundColor Green
    Write-Host " Core services operational" -ForegroundColor Green
    Write-Host " Neo4j graph database ready" -ForegroundColor Green
    Write-Host " Service connections tested" -ForegroundColor Green
    Write-Host "`nAccess URLs:" -ForegroundColor Yellow
    Write-Host "• Neo4j Browser: http://localhost:7474" -ForegroundColor Yellow
    Write-Host "• RabbitMQ Management: http://localhost:15672" -ForegroundColor Yellow
    Write-Host "• Main Application: http://localhost:5173" -ForegroundColor Yellow
    Write-Host "`nCredentials:" -ForegroundColor Yellow
    Write-Host "• Neo4j: neo4j / LegalSecure2024!" -ForegroundColor Yellow
    Write-Host "• RabbitMQ: detective / secure_password" -ForegroundColor Yellow
}
else {
    Write-Host "`n WARN:  Phase 4 Setup Incomplete" -ForegroundColor Yellow
    Write-Host "Neo4j (essential service) is not running." -ForegroundColor Yellow
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check Docker is running: docker ps" -ForegroundColor Yellow
    Write-Host "2. Start services manually: docker-compose up -d" -ForegroundColor Yellow
    Write-Host "3. Check logs: docker-compose logs neo4j" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n Ready for Phase 5: AI-Driven Real-Time UI Updates!" -ForegroundColor Magenta
