#!/bin/bash

# Evidence Processing System - Quick Start Script
echo "🚀 Evidence Processing System - Quick Start"
echo "============================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $service is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service failed to start after $max_attempts attempts"
    return 1
}

echo ""
echo "📋 Step 1: Checking Prerequisites"
echo "================================="

# Check Docker
if command_exists docker; then
    echo "✅ Docker found"
else
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Node.js
if command_exists node; then
    echo "✅ Node.js found ($(node --version))"
else
    echo "❌ Node.js not found. Please install Node.js 18+ first."
    exit 1
fi

# Check if PostgreSQL is available
if command_exists psql; then
    echo "✅ PostgreSQL client found"
else
    echo "⚠️ PostgreSQL client not found. You'll need to run the migration manually."
fi

echo ""
echo "🐳 Step 2: Starting Docker Services"
echo "===================================="

# Start RabbitMQ
echo "🐰 Starting RabbitMQ..."
docker run -d --name evidenceproc-rabbitmq \
    -p 5672:5672 -p 15672:15672 \
    -e RABBITMQ_DEFAULT_USER=evidence \
    -e RABBITMQ_DEFAULT_PASS=evidence123 \
    rabbitmq:3-management

# Start Redis
echo "🔴 Starting Redis..."
docker run -d --name evidenceproc-redis \
    -p 6379:6379 \
    redis:7

# Start Qdrant
echo "🔍 Starting Qdrant..."
docker run -d --name evidenceproc-qdrant \
    -p 6333:6333 \
    qdrant/qdrant

# Start Neo4j
echo "🕸️ Starting Neo4j..."
docker run -d --name evidenceproc-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/evidence123 \
    neo4j:5

# Start MinIO
echo "📦 Starting MinIO..."
docker run -d --name evidenceproc-minio \
    -p 9000:9000 -p 9001:9001 \
    -e MINIO_ROOT_USER=evidence \
    -e MINIO_ROOT_PASSWORD=evidence123 \
    minio/minio server /data --console-address ":9001"

echo ""
echo "⏳ Step 3: Waiting for Services"
echo "==============================="

# Wait for all services
wait_for_service "RabbitMQ" 5672
wait_for_service "Redis" 6379
wait_for_service "Qdrant" 6333
wait_for_service "Neo4j" 7687
wait_for_service "MinIO" 9000

echo ""
echo "📦 Step 4: Installing Dependencies"
echo "=================================="

# Install worker dependencies
cd workers
echo "🔧 Installing worker dependencies..."
npm install

# Install frontend dependencies (if not already done)
cd ../sveltekit-frontend
echo "🔧 Installing frontend dependencies..."
npm install uuid amqplib ioredis ws @qdrant/js-client-rest neo4j-driver minio
npm install -D @types/uuid @types/amqplib @types/ws

echo ""
echo "🗄️ Step 5: Database Setup"
echo "========================="

if command_exists psql; then
    echo "📊 Setting up PostgreSQL database..."
    echo "Please ensure your PostgreSQL database is running and accessible."
    echo "Run this command manually:"
    echo "  psql -d your_database -f ../migrations/create_evidence_processing_schema.sql"
else
    echo "⚠️ PostgreSQL client not found. Please run the migration manually:"
    echo "  psql -d your_database -f migrations/create_evidence_processing_schema.sql"
fi

echo ""
echo "⚙️ Step 6: System Setup"
echo "======================="

cd ../workers
echo "🔧 Setting up queues and brokers..."
node setup-queues.js

echo ""
echo "🏥 Step 7: Health Check"
echo "======================="

echo "🔍 Running system health check..."
node health-check.js

echo ""
echo "🎉 Step 8: Starting Worker"
echo "=========================="

echo "🚀 Starting evidence processing worker..."
echo "The worker will now listen for processing jobs."
echo ""
echo "📋 Service URLs:"
echo "  • RabbitMQ Management: http://localhost:15672 (evidence/evidence123)"
echo "  • Neo4j Browser: http://localhost:7474 (neo4j/evidence123)"
echo "  • MinIO Console: http://localhost:9001 (evidence/evidence123)"
echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "🔧 To test the system:"
echo "  1. Upload evidence files to MinIO"
echo "  2. Make POST request to /api/evidence/process"
echo "  3. Watch real-time progress in your Svelte app"
echo ""
echo "⏹️ To stop all services:"
echo "  docker stop evidenceproc-rabbitmq evidenceproc-redis evidenceproc-qdrant evidenceproc-neo4j evidenceproc-minio"
echo "  docker rm evidenceproc-rabbitmq evidenceproc-redis evidenceproc-qdrant evidenceproc-neo4j evidenceproc-minio"
echo ""

# Start the worker
npm start
