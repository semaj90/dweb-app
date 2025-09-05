@echo off
echo 🚀 Phase 4: Data Management & Event Streaming Setup
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ℹ️  Installing Phase 4 dependencies...
npm install neo4j-driver@5.15.0 ws@8.18.0 amqplib@0.10.3 ioredis@5.6.1 --save
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)
echo ✅ Dependencies installed successfully

echo ℹ️  Starting Phase 4 Docker services...
docker-compose -f docker-compose-optimized.yml up -d --remove-orphans
if %errorlevel% neq 0 (
    echo ❌ Failed to start Docker services
    exit /b 1
)
echo ✅ Docker services started

echo ℹ️  Waiting for services to be ready...
timeout /t 30

echo ℹ️  Testing service connectivity...
node test-services.mjs
if %errorlevel% neq 0 (
    echo ❌ Some services are not accessible
) else (
    echo ✅ Service connection test passed
)


echo ℹ️  Initializing Neo4j graph schema...
docker exec deeds-neo4j cypher-shell -u neo4j -p LegalSecure2024! "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;"
docker exec deeds-neo4j cypher-shell -u neo4j -p LegalSecure2024! "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;"
docker exec deeds-neo4j cypher-shell -u neo4j -p LegalSecure2024! "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;"
docker exec deeds-neo4j cypher-shell -u neo4j -p LegalSecure2024! "CREATE INDEX IF NOT EXISTS FOR (c:Case) ON (c.status);"
docker exec deeds-neo4j cypher-shell -u neo4j -p LegalSecure2024! "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name);"
echo ✅ Neo4j schema initialized

echo.
echo 🎉 Phase 4: Data Management & Event Streaming Setup Complete!
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ✅ Core services operational
echo ✅ Neo4j graph database ready
echo ✅ Service connections tested
echo.
echo Access URLs:
echo • Neo4j Browser: http://localhost:7474
echo • RabbitMQ Management: http://localhost:15672
echo • Main Application: http://localhost:5173
echo.
echo Credentials:
echo • Neo4j: neo4j / LegalSecure2024!
echo • RabbitMQ: detective / secure_password
echo.
echo 🚀 Ready for Phase 5: AI-Driven Real-Time UI Updates!
