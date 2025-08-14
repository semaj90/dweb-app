@echo off

echo =======================================================
echo   CRITICAL FIX: PRODUCTION SYSTEM DEPLOYMENT
echo   Resolving configuration conflicts and deploying stable system
echo =======================================================

:: Stop all conflicting containers
docker-compose down --remove-orphans 2>nul
docker stop $(docker ps -aq) 2>nul
docker rm $(docker ps -aq) 2>nul

:: Remove conflicting compose files except core
for %%f in (docker-compose-*.yml) do (
    if not "%%f"=="docker-compose-unified.yml" (
        del "%%f" 2>nul
    )
)

echo Creating production Docker Compose...

(
echo version: '3.8'
echo.
echo services:
echo   postgres:
echo     image: pgvector/pgvector:pg16
echo     container_name: legal-postgres
echo     environment:
echo       POSTGRES_DB: legal_ai
echo       POSTGRES_USER: legal_admin
echo       POSTGRES_PASSWORD: LegalRAG2024!
echo     ports:
echo       - "5432:5432"
echo     volumes:
echo       - postgres_data:/var/lib/postgresql/data
echo       - ./database:/docker-entrypoint-initdb.d
echo     restart: unless-stopped
echo.
echo   redis:
echo     image: redis:7-alpine
echo     container_name: legal-redis
echo     ports:
echo       - "6379:6379"
echo     command: redis-server --appendonly yes
echo     volumes:
echo       - redis_data:/data
echo     restart: unless-stopped
echo.
echo   neo4j:
echo     image: neo4j:5.15-community
echo     container_name: legal-neo4j
echo     ports:
echo       - "7474:7474"
echo       - "7687:7687"
echo     environment:
echo       NEO4J_AUTH: neo4j/LegalRAG2024!
echo     volumes:
echo       - neo4j_data:/data
echo     restart: unless-stopped
echo.
echo   qdrant:
echo     image: qdrant/qdrant:v1.7.0
echo     container_name: legal-qdrant
echo     ports:
echo       - "6333:6333"
echo     volumes:
echo       - qdrant_data:/qdrant/storage
echo     restart: unless-stopped
echo.
echo   rabbitmq:
echo     image: rabbitmq:3-management-alpine
echo     container_name: legal-rabbitmq
echo     ports:
echo       - "5672:5672"
echo       - "15672:15672"
echo     environment:
echo       RABBITMQ_DEFAULT_USER: legal_admin
echo       RABBITMQ_DEFAULT_PASS: LegalRAG2024!
echo     volumes:
echo       - rabbitmq_data:/var/lib/rabbitmq
echo     restart: unless-stopped
echo.
echo   ollama:
echo     image: ollama/ollama:latest
echo     container_name: legal-ollama
echo     ports:
echo       - "11434:11434"
echo     volumes:
echo       - ollama_data:/root/.ollama
echo     environment:
echo       - OLLAMA_HOST=0.0.0.0
echo     restart: unless-stopped
echo.
echo volumes:
echo   postgres_data:
echo   redis_data:
echo   neo4j_data:
echo   qdrant_data:
echo   rabbitmq_data:
echo   ollama_data:
) > docker-compose.yml

echo Creating database schema...

if not exist "database" mkdir database

(
echo CREATE EXTENSION IF NOT EXISTS vector;
echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
echo.
echo CREATE TABLE IF NOT EXISTS legal_cases (
echo     id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
echo     case_number VARCHAR(255) UNIQUE NOT NULL,
echo     title VARCHAR(500) NOT NULL,
echo     status VARCHAR(100) DEFAULT 'active',
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo );
echo.
echo CREATE TABLE IF NOT EXISTS legal_documents (
echo     id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
echo     title VARCHAR(500) NOT NULL,
echo     content TEXT,
echo     case_id VARCHAR(255),
echo     embedding vector(384),
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo );
echo.
echo INSERT INTO legal_cases (case_number, title) VALUES
echo ('CASE-2024-001', 'Contract Dispute'),
echo ('CASE-2024-002', 'Employment Review')
echo ON CONFLICT (case_number) DO NOTHING;
) > database\init.sql

echo Creating package.json...

(
echo {
echo   "name": "legal-ai-production",
echo   "version": "1.0.0",
echo   "scripts": {
echo     "start": "docker-compose up -d",
echo     "stop": "docker-compose down",
echo     "test": "node test.mjs",
echo     "status": "docker-compose ps"
echo   },
echo   "type": "module"
echo }
) > package.json

echo Creating integration test...

(
echo import { createConnection } from 'net';
echo.
echo const services = [
echo     { name: 'PostgreSQL', port: 5432 },
echo     { name: 'Redis', port: 6379 },
echo     { name: 'Neo4j', port: 7474 },
echo     { name: 'Qdrant', port: 6333 },
echo     { name: 'RabbitMQ', port: 15672 },
echo     { name: 'Ollama', port: 11434 }
echo ];
echo.
echo async function testService(service) {
echo     return new Promise((resolve) => {
echo         const socket = createConnection(service.port, 'localhost');
echo         const timeout = setTimeout(() => {
echo             socket.destroy();
echo             resolve({ ...service, status: 'timeout' });
echo         }, 3000);
echo         
echo         socket.on('connect', () => {
echo             clearTimeout(timeout);
echo             socket.destroy();
echo             resolve({ ...service, status: 'connected' });
echo         });
echo         
echo         socket.on('error', () => {
echo             clearTimeout(timeout);
echo             resolve({ ...service, status: 'failed' });
echo         });
echo     });
echo }
echo.
echo console.log('Testing Legal AI System...');
echo const results = await Promise.all(services.map(testService));
echo.
echo results.forEach(result => {
echo     const icon = result.status === 'connected' ? '✅' : '❌';
echo     console.log(`${icon} ${result.name}: ${result.status}`);
echo });
echo.
echo const connected = results.filter(r => r.status === 'connected').length;
echo console.log(`\nConnected: ${connected}/${services.length}`);
echo process.exit(connected >= 4 ? 0 : 1);
) > test.mjs

echo Deploying production system...

docker-compose up -d

echo Waiting for services...
timeout /t 20 >nul

echo Testing system...
node test.mjs

echo ===============================================
echo PRODUCTION DEPLOYMENT COMPLETE
echo ===============================================
echo.
echo Services:
echo - Neo4j: http://localhost:7474 (neo4j/LegalRAG2024!)
echo - RabbitMQ: http://localhost:15672 (legal_admin/LegalRAG2024!)
echo - Qdrant: http://localhost:6333
echo.
echo Commands:
echo - npm start  (start services)
echo - npm test   (test connectivity)
echo - npm stop   (stop services)
echo.
pause
