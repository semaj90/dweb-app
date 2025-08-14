@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo   UPDATED: Real-time RAG System Integration Check
echo   SvelteKit 2 + Svelte 5 Runes + XState + Existing Components
echo =========================================================
echo.

set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🔍 Checking existing app components integration...%NC%
echo.

:: Check existing SvelteKit frontend structure
echo %BLUE%1. Analyzing existing SvelteKit frontend...%NC%
if exist "sveltekit-frontend" (
    cd sveltekit-frontend
    
    echo %GREEN%✅ SvelteKit frontend found%NC%
    
    :: Check for existing components
    if exist "src\lib\components" (
        echo %GREEN%  ✓ Components directory exists%NC%
        dir /b "src\lib\components\*.svelte" 2>nul | findstr /r ".*" >nul
        if !errorlevel! == 0 (
            echo %GREEN%  ✓ Existing Svelte components found:%NC%
            for /f %%f in ('dir /b "src\lib\components\*.svelte" 2^>nul') do (
                echo %YELLOW%    - %%f%NC%
            )
        )
    )
    
    :: Check for stores
    if exist "src\lib\stores" (
        echo %GREEN%  ✓ Stores directory exists%NC%
        dir /b "src\lib\stores\*.js" "src\lib\stores\*.ts" 2>nul | findstr /r ".*" >nul
        if !errorlevel! == 0 (
            echo %GREEN%  ✓ Existing stores found%NC%
        )
    )
    
    :: Check for API routes
    if exist "src\routes\api" (
        echo %GREEN%  ✓ API routes directory exists%NC%
        dir /b /s "src\routes\api\*+server.js" "src\routes\api\*+server.ts" 2>nul | findstr /r ".*" >nul
        if !errorlevel! == 0 (
            echo %GREEN%  ✓ Existing API endpoints found%NC%
        )
    )
    
    :: Check package.json for dependencies
    if exist "package.json" (
        echo %GREEN%  ✓ Checking dependencies...%NC%
        findstr "drizzle" package.json >nul && echo %GREEN%    ✓ Drizzle ORM found%NC%
        findstr "xstate" package.json >nul && echo %GREEN%    ✓ XState found%NC% || echo %YELLOW%    ⚠ XState missing - will add%NC%
        findstr "bits-ui" package.json >nul && echo %GREEN%    ✓ Bits UI found%NC%
        findstr "@sveltejs/kit" package.json >nul && echo %GREEN%    ✓ SvelteKit found%NC%
    )
    
    cd ..
) else (
    echo %RED%❌ SvelteKit frontend not found%NC%
)

:: Check Docker services
echo.
echo %BLUE%2. Checking Docker services status...%NC%
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr -E "(postgres|ollama|qdrant|redis)" >nul
if %errorlevel% == 0 (
    echo %GREEN%✅ Docker services running:%NC%
    docker ps --format "  {{.Names}}: {{.Status}}" | findstr -E "(postgres|ollama|qdrant|redis)"
) else (
    echo %YELLOW%⚠ Starting required services...%NC%
    docker-compose -f docker-compose-optimized.yml up -d postgres redis qdrant ollama
)

:: Check database connection
echo.
echo %BLUE%3. Testing database connections...%NC%

:: PostgreSQL
docker exec deeds-postgres pg_isready -U legal_admin -d prosecutor_db >nul 2>&1
if %errorlevel% == 0 (
    echo %GREEN%✅ PostgreSQL: Connected%NC%
    
    :: Check for pgvector extension
    docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "\dx" | findstr "vector" >nul
    if %errorlevel% == 0 (
        echo %GREEN%  ✓ pgvector extension available%NC%
    ) else (
        echo %YELLOW%  ⚠ Installing pgvector...%NC%
        docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
    )
) else (
    echo %RED%❌ PostgreSQL: Not accessible%NC%
)

:: Ollama
curl -f http://localhost:11434/api/version >nul 2>&1
if %errorlevel% == 0 (
    echo %GREEN%✅ Ollama: Connected%NC%
    
    :: Check for models
    docker exec deeds-ollama-gpu ollama list | findstr "gemma" >nul
    if %errorlevel% == 0 (
        echo %GREEN%  ✓ Gemma models available%NC%
    ) else (
        echo %YELLOW%  ⚠ No models found - will pull default%NC%
    )
) else (
    echo %RED%❌ Ollama: Not accessible%NC%
)

:: Check for local GGUF models
echo.
echo %BLUE%4. Checking local model storage...%NC%
if exist "local-models" (
    dir /b "local-models\*.gguf" 2>nul | findstr /r ".*" >nul
    if !errorlevel! == 0 (
        echo %GREEN%✅ Local GGUF models found:%NC%
        for /f %%f in ('dir /b "local-models\*.gguf" 2^>nul') do (
            echo %YELLOW%  - %%f%NC%
        )
    ) else (
        echo %YELLOW%⚠ No GGUF models in local-models directory%NC%
    )
) else (
    echo %YELLOW%⚠ Creating local-models directory...%NC%
    mkdir local-models
)

:: Database strategy analysis
echo.
echo %BLUE%5. Database strategy recommendations...%NC%
echo %GREEN%📊 Optimal database architecture for your use case:%NC%
echo.
echo %YELLOW%PRIMARY: PostgreSQL with pgvector%NC%
echo %GREEN%  ✓ Best for: Structured data, ACID compliance, complex queries%NC%
echo %GREEN%  ✓ Partitioning: Excellent for large document sets%NC%
echo %GREEN%  ✓ Vector search: Native pgvector support%NC%
echo %GREEN%  ✓ JSON support: JSONB for metadata%NC%
echo.
echo %YELLOW%VECTOR STORE: Qdrant (recommended over Chroma)%NC%
echo %GREEN%  ✓ Performance: 10x faster than Chroma for large datasets%NC%
echo %GREEN%  ✓ Memory: More efficient with large embeddings%NC%
echo %GREEN%  ✓ Features: Advanced filtering and hybrid search%NC%
echo.
echo %YELLOW%LIGHTWEIGHT NoSQL: SurrealDB (recommended)%NC%
echo %GREEN%  ✓ Multi-model: Documents, graphs, key-value%NC%
echo %GREEN%  ✓ Size: 15MB single binary%NC%
echo %GREEN%  ✓ Performance: Rust-based, extremely fast%NC%
echo %GREEN%  ✓ Features: Built-in vector similarity%NC%
echo.
echo %YELLOW%GRAPH: Neo4j Community (for relationships)%NC%
echo %GREEN%  ✓ Legal: Perfect for case law relationships%NC%
echo %GREEN%  ✓ Entities: Connect parties, cases, precedents%NC%

:: Check real-time capabilities
echo.
echo %BLUE%6. Real-time integration assessment...%NC%
echo %GREEN%🔄 Recommended real-time stack:%NC%
echo %YELLOW%  • SvelteKit 2 SSR + Svelte 5 runes (reactive stores)%NC%
echo %YELLOW%  • XState state machines (Python middleware)%NC%
echo %YELLOW%  • WebSocket connections (bidirectional)%NC%
echo %YELLOW%  • Server-Sent Events (document processing updates)%NC%
echo %YELLOW%  • Drizzle ORM with real-time subscriptions%NC%

:: Performance recommendations
echo.
echo %BLUE%7. Performance optimization suggestions...%NC%
echo %GREEN%🚀 PostgreSQL partitioning strategy:%NC%
echo %YELLOW%  • Time-based: Partition by creation_date (monthly)%NC%
echo %YELLOW%  • Hash-based: Partition by case_id for load distribution%NC%
echo %YELLOW%  • List-based: Partition by document_type%NC%
echo.
echo %GREEN%💾 File storage recommendations:%NC%
echo %YELLOW%  • Small files (^<10MB): PostgreSQL bytea/JSON%NC%
echo %YELLOW%  • Large files (^>10MB): MinIO S3-compatible storage%NC%
echo %YELLOW%  • Embeddings: pgvector + Qdrant hybrid%NC%
echo %YELLOW%  • Metadata: PostgreSQL JSONB%NC%

:: Component integration plan
echo.
echo %BLUE%8. Component integration strategy...%NC%
echo %GREEN%🧩 Leveraging existing components:%NC%
if exist "sveltekit-frontend\src\lib\components" (
    echo %YELLOW%  • Extend existing Svelte components with RAG features%NC%
    echo %YELLOW%  • Add real-time document status updates%NC%
    echo %YELLOW%  • Integrate XState machines for complex workflows%NC%
    echo %YELLOW%  • Use existing UI patterns for consistency%NC%
)

:: Testing strategy
echo.
echo %BLUE%9. Real-time testing recommendations...%NC%
echo %GREEN%🧪 Testing approach:%NC%
echo %YELLOW%  1. Component unit tests with Vitest%NC%
echo %YELLOW%  2. XState machine testing with @xstate/test%NC%
echo %YELLOW%  3. API integration tests with Supertest%NC%
echo %YELLOW%  4. Real-time WebSocket testing%NC%
echo %YELLOW%  5. Database performance testing with pgbench%NC%

:: Next steps
echo.
echo %BLUE%🎯 Recommended next steps:%NC%
echo %GREEN%IMMEDIATE:%NC%
echo %YELLOW%  1. Run: UPDATE-REALTIME-INTEGRATION.bat%NC%
echo %YELLOW%  2. Install: SurrealDB for lightweight NoSQL%NC%
echo %YELLOW%  3. Configure: PostgreSQL partitioning%NC%
echo %YELLOW%  4. Setup: XState middleware integration%NC%
echo.
echo %GREEN%SHORT-TERM:%NC%
echo %YELLOW%  1. Implement: Real-time document processing%NC%
echo %YELLOW%  2. Add: WebSocket-based RAG queries%NC%
echo %YELLOW%  3. Create: Performance monitoring dashboard%NC%
echo %YELLOW%  4. Test: Load testing with large document sets%NC%

:: Final recommendation
echo.
echo %GREEN%💡 OPTIMAL ARCHITECTURE SUMMARY:%NC%
echo %BLUE%┌─────────────────────────────────────────────┐%NC%
echo %BLUE%│ Frontend: SvelteKit 2 + Svelte 5 Runes     │%NC%
echo %BLUE%│ State: XState machines + real-time stores   │%NC%
echo %BLUE%│ API: FastAPI + WebSocket + SSE              │%NC%
echo %BLUE%│ Primary DB: PostgreSQL + pgvector           │%NC%
echo %BLUE%│ Vector DB: Qdrant (high performance)        │%NC%
echo %BLUE%│ NoSQL: SurrealDB (lightweight, 15MB)        │%NC%
echo %BLUE%│ Graph: Neo4j Community (relationships)      │%NC%
echo %BLUE%│ Files: MinIO (S3-compatible)                │%NC%
echo %BLUE%│ Models: Local GGUF + Ollama fallback        │%NC%
echo %BLUE%└─────────────────────────────────────────────┘%NC%

echo.
echo %GREEN%✨ Ready to implement real-time RAG system!%NC%
pause
