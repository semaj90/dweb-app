@echo off
echo ===========================================
echo  PostgreSQL-First Legal AI System Startup
echo ===========================================

echo.
echo 1. Starting PostgreSQL...
net start postgresql-17

echo.
echo 2. Starting Redis...
start /B "" "C:\Users\james\Desktop\deeds-web\deeds-web-app\redis-windows\redis-server.exe"

echo.
echo 3. Starting Qdrant...
start /B "" "C:\Users\james\Desktop\deeds-web\deeds-web-app\qdrant-windows\qdrant.exe"

echo.
echo 4. Starting Ollama...
start /B "" ollama serve

echo.
echo 5. Waiting for services to initialize...
timeout /t 10 /nobreak >nul

echo.
echo 6. Running database migrations...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -f "drizzle\20250822_defaults_and_vector.sql"
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -f "drizzle\20250822_vectors_autocreate_and_notify.sql"

echo.
echo 7. Starting PG-Redis relay service...
start /B "" tsx src\lib\server\relays\pg-redis-relay.ts

echo.
echo 8. Starting Go microservice...
start /B "" go run simple-api-endpoints.go

echo.
echo 9. Starting SvelteKit development server...
npm run dev

echo.
echo ===========================================
echo  All services started!
echo  PostgreSQL-first architecture is running
echo ===========================================
echo.
echo Access points:
echo - Frontend: http://localhost:5173
echo - Go Service: http://localhost:8080
echo - Ollama: http://localhost:11434
echo - Qdrant: http://localhost:6333
echo.
pause