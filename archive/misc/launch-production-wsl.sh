# WSL-compatible production launch script
#!/bin/bash
echo "🚀 Phase 5: WSL Production Launch"
echo "================================"

# Check WSL environment
if [[ ! -d "/mnt/c" ]]; then
    echo "❌ Not running in WSL"
    exit 1
fi

# Navigate to Windows project directory via WSL
cd "/mnt/c/Users/james/Desktop/deeds-web/deeds-web-app"

# Check Docker Desktop integration
if ! docker version &>/dev/null; then
    echo "❌ Docker not available in WSL"
    echo "Enable WSL integration in Docker Desktop"
    exit 1
fi

echo "Starting production environment..."

# Set production environment
export NODE_ENV=production
export POSTGRES_PASSWORD="LegalAI2024!Prod"

# Start production stack
docker-compose -f deployment/production/docker-compose.prod.yml up -d

# Wait for services
sleep 30

# Health checks
echo "Checking service health..."
docker exec legal-ai-postgres-prod pg_isready -U legal_admin
docker exec legal-ai-redis-prod redis-cli ping
curl -s http://localhost:6333/health

echo "✅ Production environment ready!"
echo "Access: https://localhost"
echo "Logs: docker-compose -f deployment/production/docker-compose.prod.yml logs -f"
