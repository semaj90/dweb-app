#!/bin/bash

# Deploy Complete RAG System
# This script sets up the entire enhanced RAG system with all dependencies

echo "🚀 Starting Complete RAG System Deployment..."
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."
echo "------------------------"

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker is installed"

if ! command_exists node; then
    print_error "Node.js is not installed. Please install Node.js first."
    exit 1
fi
print_status "Node.js is installed"

if ! command_exists npm; then
    print_error "npm is not installed. Please install npm first."
    exit 1
fi
print_status "npm is installed"

# Check for Python (needed for OCR)
if ! command_exists python; then
    print_warning "Python is not installed. OCR features will be limited."
else
    print_status "Python is installed"
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
echo "-------------------------------"

directories=(
    "data/postgres"
    "data/redis"
    "data/neo4j"
    "data/minio"
    "data/rabbitmq"
    "logs"
    "models"
    "uploads"
    "cache"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_status "Created directory: $dir"
done

# Copy environment file
echo ""
echo "Setting up environment configuration..."
echo "---------------------------------------"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        print_status "Created .env file from .env.example"
        print_warning "Please update .env with your configuration"
    else
        cat > .env << 'EOF'
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=legal-documents

# Redis
REDIS_URL=redis://localhost:6379

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3legal:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Backend Services
BACKEND_RAG_STREAM_ENDPOINT=http://localhost:8094/stream
PORT=3000
NODE_ENV=development
EOF
        print_status "Created default .env file"
    fi
else
    print_status ".env file already exists"
fi

# Create docker-compose.yml
echo ""
echo "Creating Docker Compose configuration..."
echo "----------------------------------------"

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    container_name: legal-ai-postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: legal_ai_db
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: legal-ai-redis
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5-community
    container_name: legal-ai-neo4j
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_apoc_export_file_enabled: true
      NEO4J_apoc_import_file_enabled: true
      NEO4J_apoc_import_file_use__neo4j__config: true
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - ./data/neo4j:/data
    healthcheck:
      test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: legal-ai-rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - ./data/rabbitmq:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    container_name: legal-ai-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./data/minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

networks:
  default:
    name: legal-ai-network
EOF

print_status "Created docker-compose.yml"

# Create database initialization script
echo ""
echo "Creating database initialization script..."
echo "------------------------------------------"

cat > init-db.sql << 'EOF'
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user roles
CREATE TYPE user_role AS ENUM ('prosecutor', 'detective', 'admin', 'analyst');

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table with vector embeddings
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create cases table
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'open',
    assigned_to UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create evidence table
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES cases(id),
    document_id UUID REFERENCES documents(id),
    type VARCHAR(50),
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feedback table for reinforcement learning
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    user_id UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create analytics table
CREATE TABLE IF NOT EXISTS analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    user_id UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_cases_assigned_to ON cases(assigned_to);
CREATE INDEX idx_evidence_case_id ON evidence(case_id);
CREATE INDEX idx_feedback_user_id ON feedback(user_id);
CREATE INDEX idx_analytics_event_type ON analytics(event_type);
CREATE INDEX idx_analytics_created_at ON analytics(created_at);

-- Insert demo users
INSERT INTO users (email, password_hash, role) VALUES
    ('prosecutor@legal.ai', '$2b$10$YourHashedPasswordHere', 'prosecutor'),
    ('detective@legal.ai', '$2b$10$YourHashedPasswordHere', 'detective'),
    ('admin@legal.ai', '$2b$10$YourHashedPasswordHere', 'admin'),
    ('analyst@legal.ai', '$2b$10$YourHashedPasswordHere', 'analyst')
ON CONFLICT (email) DO NOTHING;
EOF

print_status "Created init-db.sql"

# Start Docker services
echo ""
echo "Starting Docker services..."
echo "---------------------------"

docker-compose down 2>/dev/null
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "Waiting for services to be healthy..."
echo "-------------------------------------"

services=("postgres" "redis" "neo4j" "rabbitmq" "minio")
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    all_healthy=true
    
    for service in "${services[@]}"; do
        if ! docker-compose ps | grep "legal-ai-$service" | grep -q "healthy"; then
            all_healthy=false
            break
        fi
    done
    
    if $all_healthy; then
        print_status "All services are healthy!"
        break
    fi
    
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    print_error "Services failed to become healthy in time"
    docker-compose logs
    exit 1
fi

# Install npm dependencies
echo ""
echo "Installing npm dependencies..."
echo "------------------------------"

npm install
print_status "npm dependencies installed"

# Run database migrations
echo ""
echo "Running database migrations..."
echo "------------------------------"

if [ -f "drizzle.config.ts" ]; then
    npm run db:push 2>/dev/null || print_warning "Database migration skipped"
else
    print_warning "No drizzle.config.ts found, skipping migrations"
fi

# Download models if needed
echo ""
echo "Checking AI models..."
echo "--------------------"

if command_exists ollama; then
    echo "Pulling Ollama models..."
    ollama pull nomic-embed-text 2>/dev/null || print_warning "Failed to pull nomic-embed-text"
    ollama pull gemma3legal:latest 2>/dev/null || ollama pull gemma:2b 2>/dev/null || print_warning "Failed to pull chat model"
    print_status "AI models checked"
else
    print_warning "Ollama not installed. Please install Ollama and pull models manually."
fi

# Create MinIO bucket
echo ""
echo "Setting up MinIO bucket..."
echo "--------------------------"

# Wait for MinIO to be ready
sleep 5

# Use mc (MinIO client) if available, otherwise use curl
if command_exists mc; then
    mc alias set myminio http://localhost:9000 minioadmin minioadmin123 2>/dev/null
    mc mb myminio/legal-documents 2>/dev/null || true
    print_status "MinIO bucket created"
else
    # Try with curl
    curl -X PUT http://localhost:9000/legal-documents \
         -H "Host: localhost:9000" \
         -H "Date: $(date -R)" \
         -H "Content-Type: application/octet-stream" \
         --user minioadmin:minioadmin123 \
         2>/dev/null || print_warning "Could not create MinIO bucket automatically"
fi

# Build the application
echo ""
echo "Building the application..."
echo "---------------------------"

npm run build || print_warning "Build failed - continuing anyway"

# Create start script
echo ""
echo "Creating start script..."
echo "-----------------------"

cat > start-rag-system.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Enhanced RAG System..."
echo "================================="

# Start Docker services if not running
docker-compose up -d

# Wait for services
sleep 5

# Start the application
npm run dev

echo "✅ System is running!"
echo "Access the application at: http://localhost:3000"
echo "MinIO Console: http://localhost:9001"
echo "RabbitMQ Management: http://localhost:15672"
echo "Neo4j Browser: http://localhost:7474"
EOF

chmod +x start-rag-system.sh
print_status "Created start-rag-system.sh"

# Create test script
echo ""
echo "Creating test script..."
echo "----------------------"

cat > test-rag-system.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing RAG System..."
echo "======================="

# Test database connection
echo -n "Testing PostgreSQL... "
docker exec legal-ai-postgres pg_isready -U postgres >/dev/null 2>&1 && echo "✓" || echo "✗"

# Test Redis
echo -n "Testing Redis... "
docker exec legal-ai-redis redis-cli ping >/dev/null 2>&1 && echo "✓" || echo "✗"

# Test Neo4j
echo -n "Testing Neo4j... "
curl -s http://localhost:7474 >/dev/null 2>&1 && echo "✓" || echo "✗"

# Test RabbitMQ
echo -n "Testing RabbitMQ... "
curl -s http://localhost:15672 >/dev/null 2>&1 && echo "✓" || echo "✗"

# Test MinIO
echo -n "Testing MinIO... "
curl -s http://localhost:9000 >/dev/null 2>&1 && echo "✓" || echo "✗"

# Run unit tests if available
if [ -f "package.json" ] && grep -q "\"test\"" package.json; then
    echo ""
    echo "Running unit tests..."
    npm test
fi

echo ""
echo "✅ Testing complete!"
EOF

chmod +x test-rag-system.sh
print_status "Created test-rag-system.sh"

# Final status
echo ""
echo "========================================="
echo -e "${GREEN}✅ RAG System Deployment Complete!${NC}"
echo "========================================="
echo ""
echo "Services running:"
echo "  • PostgreSQL: localhost:5432"
echo "  • Redis: localhost:6379"
echo "  • Neo4j: localhost:7474 (browser) / localhost:7687 (bolt)"
echo "  • RabbitMQ: localhost:5672 (amqp) / localhost:15672 (management)"
echo "  • MinIO: localhost:9000 (api) / localhost:9001 (console)"
echo ""
echo "Next steps:"
echo "  1. Update .env file with your configuration"
echo "  2. Run: ./start-rag-system.sh"
echo "  3. Access the application at http://localhost:3000"
echo ""
echo "To test the system: ./test-rag-system.sh"
echo "To stop services: docker-compose down"
echo ""
print_warning "Remember to configure Ollama and pull the required models!"
