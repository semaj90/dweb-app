#!/bin/bash

# Legal AI Assistant - WSL Docker Desktop Startup Script
# Run this from WSL to start your full AI stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Legal AI Assistant - WSL Docker Setup${NC}"
echo -e "${BLUE}===========================================${NC}"
echo

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker Desktop is running
echo -e "${BLUE}🔍 Checking Docker Desktop...${NC}"
if ! docker info >/dev/null 2>&1; then
    print_error "Docker Desktop is not running"
    echo "Please start Docker Desktop and enable WSL integration"
    echo "Settings > Resources > WSL Integration > Enable integration with additional distros"
    exit 1
fi
print_status "Docker Desktop is running"

# Check if WSL integration is working
echo -e "${BLUE}🔍 Checking WSL integration...${NC}"
if ! docker ps >/dev/null 2>&1; then
    print_error "Docker WSL integration not working"
    echo "Enable WSL integration in Docker Desktop settings"
    exit 1
fi
print_status "WSL integration is working"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose not found"
    echo "Install docker-compose or use 'docker compose' (newer version)"
    exit 1
fi
print_status "Docker Compose is available"

# Set project directory (convert Windows path to WSL path if needed)
PROJECT_DIR="$(pwd)"
print_status "Working in: $PROJECT_DIR"

# Stop any existing containers
echo -e "${BLUE}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose-fixed.yml down --remove-orphans 2>/dev/null || true

# Pull latest images
echo -e "${BLUE}⬇️  Pulling Docker images...${NC}"
docker-compose -f docker-compose-fixed.yml pull

# Build custom images
echo -e "${BLUE}🔨 Building custom images...${NC}"
docker-compose -f docker-compose-fixed.yml build

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"
docker-compose -f docker-compose-fixed.yml up -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
sleep 30

# Check service health
echo -e "${BLUE}🏥 Checking service health...${NC}"
services=(
    "PostgreSQL:5432"
    "Ollama:11434" 
    "Qdrant:6333"
    "Neo4j:7474"
    "RabbitMQ:15672"
    "Redis:6379"
    "SvelteKit:5173"
)

all_healthy=true

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if nc -z localhost $port 2>/dev/null; then
        print_status "$name is responding on port $port"
    else
        print_warning "$name is not responding on port $port"
        all_healthy=false
    fi
done

echo
if [ "$all_healthy" = true ]; then
    print_status "All services are healthy!"
    echo
    echo -e "${GREEN}🎉 Your Legal AI stack is ready!${NC}"
    echo
    echo -e "${BLUE}📱 Access points:${NC}"
    echo "• SvelteKit App:     http://localhost:5173"
    echo "• Neo4j Browser:     http://localhost:7474"
    echo "• RabbitMQ Mgmt:     http://localhost:15672"
    echo "• Qdrant Dashboard:  http://localhost:6333/dashboard"
    echo
    echo -e "${BLUE}🔧 Management commands:${NC}"
    echo "• View logs:         npm run docker:logs"
    echo "• Check status:      npm run docker:status" 
    echo "• Health check:      npm run health"
    echo "• Stop services:     npm run docker:down"
    echo
    echo -e "${BLUE}🤖 AI Model Setup:${NC}"
    echo "• Load Gemma3 model: docker exec deeds-ollama-gpu ollama pull gemma3"
    echo "• Test AI:           curl -X POST http://localhost:11434/api/generate -d '{\"model\":\"gemma3\",\"prompt\":\"Hello\",\"stream\":false}'"
else
    print_warning "Some services are not healthy. Check logs with: docker-compose -f docker-compose-fixed.yml logs"
fi

echo
echo -e "${BLUE}💡 Pro tip: Use VS Code with the Remote-WSL extension for the best development experience!${NC}"
