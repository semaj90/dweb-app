@echo off
echo 🚀 Enhanced Legal AI System Setup - Complete Edition
echo =====================================================

echo.
echo 📚 Generating comprehensive best practices documentation...

REM Generate the best practices guide
echo # Legal AI System Best Practices Guide > BEST_PRACTICES_COMPREHENSIVE.md
echo Generated: %DATE% %TIME% >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ## 🔐 Security Best Practices >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Database Security >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Use strong passwords (16+ characters with mixed case, numbers, symbols) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Enable SSL/TLS for database connections >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Implement connection pooling with limits >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Regular security audits and updates >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Backup encryption with rotation >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### API Security >> BEST_PRACTICES_COMPREHENSIVE.md
echo - JWT tokens with short expiration (1 hour) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Rate limiting: 100 requests/minute per IP >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Input validation and sanitization >> BEST_PRACTICES_COMPREHENSIVE.md
echo - CORS configuration for specific origins >> BEST_PRACTICES_COMPREHENSIVE.md
echo - API key rotation every 90 days >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ## ⚡ Performance Best Practices >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Database Optimization >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Index all foreign keys and search columns >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Use prepared statements for all queries >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Connection pooling: min 5, max 25 connections >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Query timeout: 30 seconds >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Regular VACUUM and ANALYZE >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Caching Strategy >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Redis for session data (TTL: 1 hour) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Qdrant for vector embeddings (persistent) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Application cache for API responses (TTL: 5 minutes) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - CDN for static assets >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Browser caching headers >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ## 🧠 AI/ML Best Practices >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Model Selection >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Use quantized models (Q4_K_M) for production >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Fallback chain: Local LLM to Claude to OpenAI >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Context window optimization (4K chunks) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Embedding dimension: 384 for balance >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Model warm-up on service start >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Vector Search >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Cosine similarity for semantic search >> BEST_PRACTICES_COMPREHENSIVE.md
echo - HNSW index for fast nearest neighbor >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Batch processing for bulk operations >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Result caching for common queries >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Relevance scoring threshold: 0.7 >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ## 🚀 Development Best Practices >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### SvelteKit 2 + Svelte 5 >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Use $state() for reactive variables >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Use $derived() for computed values >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Use $effect() for side effects >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Progressive enhancement with use:enhance >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Type safety with generated types >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Go Microservices >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Structured logging with levels >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Graceful shutdown handling >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Health check endpoints >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Circuit breaker pattern >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Metrics collection (Prometheus) >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ## 📊 Monitoring Best Practices >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo ### Metrics Collection >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Response times (p50, p95, p99) >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Error rates by endpoint >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Database connection pool usage >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Memory and CPU utilization >> BEST_PRACTICES_COMPREHENSIVE.md
echo - Cache hit/miss ratios >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo --- >> BEST_PRACTICES_COMPREHENSIVE.md
echo. >> BEST_PRACTICES_COMPREHENSIVE.md
echo This document should be reviewed and updated quarterly. >> BEST_PRACTICES_COMPREHENSIVE.md

echo ✅ Best practices guide generated: BEST_PRACTICES_COMPREHENSIVE.md

echo.
echo 🔐 Generating secure configuration template...

REM Generate secure .env template
echo # Enhanced Secure Configuration Template > .env.secure.template
echo # Generated: %DATE% %TIME% >> .env.secure.template
echo. >> .env.secure.template
echo # Database Configuration >> .env.secure.template
echo DATABASE_URL=postgresql://legal_admin:CHANGE_THIS_PASSWORD@localhost:5432/legal_ai_db >> .env.secure.template
echo POSTGRES_PASSWORD=CHANGE_THIS_PASSWORD >> .env.secure.template
echo DB_HOST=localhost >> .env.secure.template
echo DB_PORT=5432 >> .env.secure.template
echo DB_NAME=legal_ai_db >> .env.secure.template
echo DB_USER=legal_admin >> .env.secure.template
echo. >> .env.secure.template
echo # MinIO Configuration >> .env.secure.template
echo MINIO_ROOT_USER=minioadmin >> .env.secure.template
echo MINIO_ROOT_PASSWORD=CHANGE_THIS_PASSWORD >> .env.secure.template
echo MINIO_ENDPOINT=localhost:9000 >> .env.secure.template
echo. >> .env.secure.template
echo # API Configuration >> .env.secure.template
echo JWT_SECRET=CHANGE_THIS_TO_RANDOM_256_BIT_SECRET >> .env.secure.template
echo API_KEY=CHANGE_THIS_TO_YOUR_API_KEY >> .env.secure.template
echo CLAUDE_API_KEY=CHANGE_THIS_TO_YOUR_CLAUDE_KEY >> .env.secure.template
echo OPENAI_API_KEY=CHANGE_THIS_TO_YOUR_OPENAI_KEY >> .env.secure.template
echo. >> .env.secure.template
echo # Service Ports >> .env.secure.template
echo POSTGRES_PORT=5432 >> .env.secure.template
echo REDIS_PORT=6379 >> .env.secure.template
echo QDRANT_PORT=6333 >> .env.secure.template
echo MINIO_PORT=9000 >> .env.secure.template
echo GO_SERVICE_PORT=8093 >> .env.secure.template
echo SVELTEKIT_PORT=5173 >> .env.secure.template
echo. >> .env.secure.template
echo # Security >> .env.secure.template
echo CORS_ORIGIN=http://localhost:5173 >> .env.secure.template
echo SECURE_COOKIES=true >> .env.secure.template
echo SESSION_TIMEOUT=3600 >> .env.secure.template

echo ✅ Secure configuration template generated: .env.secure.template

echo.
echo 📊 Generating system monitoring script...

REM Generate monitoring script
echo @echo off > system-status-monitor.bat
echo echo === Legal AI System Status === >> system-status-monitor.bat
echo echo. >> system-status-monitor.bat
echo echo Services: >> system-status-monitor.bat
echo netstat -an ^| findstr ":5432" ^>nul 2^>^&1 ^&^& echo   ✅ PostgreSQL (5432) ^|^| echo   ❌ PostgreSQL (5432) >> system-status-monitor.bat
echo netstat -an ^| findstr ":6379" ^>nul 2^>^&1 ^&^& echo   ✅ Redis (6379) ^|^| echo   ❌ Redis (6379) >> system-status-monitor.bat
echo netstat -an ^| findstr ":6333" ^>nul 2^>^&1 ^&^& echo   ✅ Qdrant (6333) ^|^| echo   ❌ Qdrant (6333) >> system-status-monitor.bat
echo netstat -an ^| findstr ":9000" ^>nul 2^>^&1 ^&^& echo   ✅ MinIO (9000) ^|^| echo   ❌ MinIO (9000) >> system-status-monitor.bat
echo netstat -an ^| findstr ":8093" ^>nul 2^>^&1 ^&^& echo   ✅ Go Service (8093) ^|^| echo   ❌ Go Service (8093) >> system-status-monitor.bat
echo netstat -an ^| findstr ":5173" ^>nul 2^>^&1 ^&^& echo   ✅ SvelteKit (5173) ^|^| echo   ❌ SvelteKit (5173) >> system-status-monitor.bat
echo echo. >> system-status-monitor.bat
echo echo Quick Links: >> system-status-monitor.bat
echo echo   SvelteKit App: http://localhost:5173 >> system-status-monitor.bat
echo echo   MinIO Console: http://localhost:9001 >> system-status-monitor.bat
echo echo   Qdrant Dashboard: http://localhost:6333/dashboard >> system-status-monitor.bat
echo echo. >> system-status-monitor.bat
echo pause >> system-status-monitor.bat

echo ✅ System monitoring script generated: system-status-monitor.bat

echo.
echo 🚀 Testing system services...

REM Test PostgreSQL
echo Testing PostgreSQL connection...
set PGPASSWORD=123456
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" >nul 2>&1
if %errorlevel%==0 (
    echo ✅ PostgreSQL connected successfully
) else (
    echo ⚠️  PostgreSQL connection failed
)

echo.
echo 💾 Creating system backup...
set backup_dir=backup_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
mkdir "%backup_dir%" 2>nul
copy .env* "%backup_dir%\" >nul 2>&1
copy package.json "%backup_dir%\" >nul 2>&1
copy svelte.config.js "%backup_dir%\" >nul 2>&1
echo ✅ Backup created in %backup_dir%

echo.
echo ========================================
echo 🎉 Enhanced Setup Complete!
echo ========================================
echo.
echo 📄 Files generated:
echo   - BEST_PRACTICES_COMPREHENSIVE.md
echo   - .env.secure.template
echo   - system-status-monitor.bat
echo   - Backup: %backup_dir%
echo.
echo 🌐 Quick access:
echo   - SvelteKit App: http://localhost:5173
echo   - MinIO Console: http://localhost:9001
echo   - Qdrant Dashboard: http://localhost:6333/dashboard
echo.
echo 📊 Monitor system: system-status-monitor.bat
echo 🔐 Update .env.secure.template with your API keys
echo.
echo ✅ Enhanced Legal AI System is ready! 🚀
echo.

pause