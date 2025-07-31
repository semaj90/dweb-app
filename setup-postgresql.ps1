# PostgreSQL Setup Script for Enhanced RAG System
# Supports both Docker and native PostgreSQL installation

param(
    [switch]$UseDocker = $true,
    [switch]$SkipData = $false,
    [string]$Password = "password",
    [int]$Port = 5432
)

Write-Host "🔧 Setting up PostgreSQL for Enhanced RAG System..." -ForegroundColor Cyan

if ($UseDocker) {
    Write-Host "📦 Using Docker PostgreSQL setup..." -ForegroundColor Yellow
    
    # Check if Docker is available
    try {
        docker --version | Out-Null
        Write-Host "✅ Docker is available" -ForegroundColor Green
    } catch {
        Write-Host "❌ Docker not found. Please install Docker or use -UseDocker:`$false" -ForegroundColor Red
        exit 1
    }
    
    # Stop existing postgres container if it exists
    Write-Host "🛑 Stopping existing PostgreSQL container (if any)..." -ForegroundColor Yellow
    docker stop postgres 2>$null
    docker rm postgres 2>$null
    
    # Start PostgreSQL with pgvector extension
    Write-Host "🚀 Starting PostgreSQL container with pgvector..." -ForegroundColor Green
    $dockerCommand = @"
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=$Password \
  -e POSTGRES_DB=legal_ai_db \
  -e POSTGRES_USER=postgres \
  -p $Port`:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg15
"@
    
    Invoke-Expression $dockerCommand
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PostgreSQL container started successfully" -ForegroundColor Green
        
        # Wait for PostgreSQL to be ready
        Write-Host "⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
        $maxAttempts = 30
        $attempt = 0
        
        do {
            Start-Sleep -Seconds 2
            $attempt++
            $ready = docker exec postgres pg_isready -U postgres 2>$null
            
            if ($ready -match "accepting connections") {
                Write-Host "✅ PostgreSQL is ready!" -ForegroundColor Green
                break
            }
            
            Write-Host "⏳ Attempt $attempt/$maxAttempts - PostgreSQL not ready yet..." -ForegroundColor Yellow
        } while ($attempt -lt $maxAttempts)
        
        if ($attempt -ge $maxAttempts) {
            Write-Host "❌ PostgreSQL failed to start within timeout" -ForegroundColor Red
            exit 1
        }
        
    } else {
        Write-Host "❌ Failed to start PostgreSQL container" -ForegroundColor Red
        exit 1
    }
    
    # Create database and enable extensions
    Write-Host "🔧 Setting up database and extensions..." -ForegroundColor Cyan
    
    $setupSQL = @'
-- Create database if not exists
SELECT 'CREATE DATABASE legal_ai_db' 
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'legal_ai_db');

-- Connect to database and enable extensions
\c legal_ai_db

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable other useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create basic schema for legal AI
CREATE SCHEMA IF NOT EXISTS legal_ai;
CREATE SCHEMA IF NOT EXISTS vector_store;

-- Test vector functionality
SELECT vector_dims('[1,2,3]'::vector) as vector_test;

-- Show enabled extensions
SELECT name, default_version, installed_version 
FROM pg_available_extensions 
WHERE installed_version IS NOT NULL
ORDER BY name;
'@
    
    # Write SQL to temporary file
    $setupSQL | Out-File -FilePath "temp_setup.sql" -Encoding UTF8
    
    # Execute setup SQL
    docker exec -i postgres psql -U postgres -f - < temp_setup.sql
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database setup completed successfully" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Database setup completed with warnings" -ForegroundColor Yellow
    }
    
    # Clean up temporary file
    Remove-Item "temp_setup.sql" -ErrorAction SilentlyContinue
    
} else {
    Write-Host "🏠 Using native PostgreSQL setup..." -ForegroundColor Yellow
    Write-Host "ℹ️ This requires PostgreSQL to be installed and running locally" -ForegroundColor Cyan
    
    # Test native PostgreSQL connection
    try {
        $env:PGPASSWORD = $Password
        $testResult = psql -h localhost -p $Port -U postgres -d postgres -c "SELECT version();" 2>&1
        
        if ($testResult -match "PostgreSQL") {
            Write-Host "✅ Native PostgreSQL connection successful" -ForegroundColor Green
        } else {
            throw "Connection failed: $testResult"
        }
    } catch {
        Write-Host "❌ Failed to connect to native PostgreSQL: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "📋 To install PostgreSQL:" -ForegroundColor Yellow
        Write-Host "   1. Download from: https://www.postgresql.org/download/" -ForegroundColor White
        Write-Host "   2. Or use Chocolatey: choco install postgresql" -ForegroundColor White
        Write-Host "   3. Or use Scoop: scoop install postgresql" -ForegroundColor White
        exit 1
    }
}

# Test final connection
Write-Host "🧪 Testing final database connection..." -ForegroundColor Cyan

$connectionString = "postgresql://postgres:$Password@localhost:$Port/legal_ai_db"

try {
    # Create a simple Node.js test script
    $testScript = @"
import pg from 'pg';
const client = new pg.Client('$connectionString');
try {
    await client.connect();
    const result = await client.query('SELECT NOW() as current_time, version() as postgres_version');
    console.log('✅ Database connection successful!');
    console.log('🕐 Current time:', result.rows[0].current_time);
    console.log('📊 PostgreSQL version:', result.rows[0].postgres_version.split(' ')[0] + ' ' + result.rows[0].postgres_version.split(' ')[1]);
    
    // Test pgvector
    const vectorTest = await client.query("SELECT '[1,2,3]'::vector as test_vector");
    console.log('🔢 Vector extension test:', vectorTest.rows[0].test_vector);
    
    await client.end();
    process.exit(0);
} catch (error) {
    console.error('❌ Database connection failed:', error.message);
    process.exit(1);
}
"@
    
    $testScript | Out-File -FilePath "test_db_connection.mjs" -Encoding UTF8
    
    # Run the test
    node test_db_connection.mjs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🎉 PostgreSQL setup completed successfully!" -ForegroundColor Green
        Write-Host "📋 Connection details:" -ForegroundColor Cyan
        Write-Host "   Host: localhost" -ForegroundColor White
        Write-Host "   Port: $Port" -ForegroundColor White
        Write-Host "   Database: legal_ai_db" -ForegroundColor White
        Write-Host "   Username: postgres" -ForegroundColor White
        Write-Host "   Password: $Password" -ForegroundColor White
        Write-Host "   Connection String: $connectionString" -ForegroundColor White
    } else {
        Write-Host "❌ Database connection test failed" -ForegroundColor Red
    }
    
    # Clean up test file
    Remove-Item "test_db_connection.mjs" -ErrorAction SilentlyContinue
    
} catch {
    Write-Host "❌ Error during database setup: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "🏁 PostgreSQL setup script completed!" -ForegroundColor Green
Write-Host "💡 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Update your .env file with the connection string" -ForegroundColor White
Write-Host "   2. Run your Enhanced RAG System" -ForegroundColor White
Write-Host "   3. Test the vector search functionality" -ForegroundColor White