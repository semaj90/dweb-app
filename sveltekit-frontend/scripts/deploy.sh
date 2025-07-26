#!/bin/bash
set -e

echo "🚀 Starting production deployment..."

# Build and deploy
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Run migrations
docker-compose -f docker-compose.prod.yml exec app npm run db:migrate

# Health check
echo "🔍 Running health check..."
curl -f http://localhost:3000/api/health || exit 1

echo "✅ Deployment successful!"
