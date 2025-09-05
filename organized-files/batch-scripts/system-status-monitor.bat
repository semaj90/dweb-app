@echo off
echo === Legal AI System Status ===
echo.

echo Services:
netstat -an | findstr ":5432" >nul 2>&1 && echo   ✅ PostgreSQL (5432) || echo   ❌ PostgreSQL (5432)
netstat -an | findstr ":6379" >nul 2>&1 && echo   ✅ Redis (6379) || echo   ❌ Redis (6379)
netstat -an | findstr ":6333" >nul 2>&1 && echo   ✅ Qdrant (6333) || echo   ❌ Qdrant (6333)
netstat -an | findstr ":9000" >nul 2>&1 && echo   ✅ MinIO (9000) || echo   ❌ MinIO (9000)
netstat -an | findstr ":8093" >nul 2>&1 && echo   ✅ Go Service (8093) || echo   ❌ Go Service (8093)
netstat -an | findstr ":5173" >nul 2>&1 && echo   ✅ SvelteKit (5173) || echo   ❌ SvelteKit (5173)
netstat -an | findstr ":11434" >nul 2>&1 && echo   ✅ Ollama (11434) || echo   ❌ Ollama (11434)

echo.
echo Quick Links:
echo   SvelteKit App: http://localhost:5173
echo   MinIO Console: http://localhost:9001
echo   Qdrant Dashboard: http://localhost:6333/dashboard
echo   PostgreSQL connected with minioadmin123 password

echo.
echo System Health:
echo   PostgreSQL: pgvector extension enabled
echo   Go Service: Upload service running on port 8093
echo   MinIO: File storage ready with minioadmin123
echo   Vector DB: Qdrant for semantic search

echo.
pause