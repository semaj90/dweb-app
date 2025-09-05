@echo off
REM PostgreSQL Command Line Shortcuts for Legal AI System

set PGPASSWORD=123456
set PGUSER=postgres
set PGDATABASE=legal_ai_db
set PGHOST=localhost
set PGPORT=5432

IF "%1"=="connect" (
    echo Connecting to legal_ai_db...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe"
    exit /b
)

IF "%1"=="test-vector" (
    echo Testing pgvector...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -c "SELECT '[1,2,3]'::vector;"
    exit /b
)

IF "%1"=="list-docs" (
    echo Listing documents with embeddings...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -c "SELECT id, file, substring(content, 1, 50) as preview, created_at FROM documents ORDER BY created_at DESC LIMIT 10;"
    exit /b
)

IF "%1"=="search" (
    IF "%2"=="" (
        echo Error: Please provide a search query
        echo Usage: psql-shortcuts.bat search "your query here"
        exit /b 1
    )
    echo Searching for: %2
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -c "SELECT file, content FROM documents WHERE content ILIKE '%%%2%%' LIMIT 5;"
    exit /b
)

IF "%1"=="vector-status" (
    echo Checking pgvector status...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -c "SELECT COUNT(*) as total_docs, COUNT(embedding) as docs_with_embeddings FROM documents;"
    exit /b
)

IF "%1"=="run-sql" (
    IF "%2"=="" (
        echo Error: Please provide SQL file path
        echo Usage: psql-shortcuts.bat run-sql "path\to\file.sql"
        exit /b 1
    )
    echo Running SQL file: %2
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -f %2
    exit /b
)

echo PostgreSQL Shortcuts for Legal AI System
echo ========================================
echo Usage: psql-shortcuts.bat [command] [args]
echo.
echo Commands:
echo   connect        - Connect to legal_ai_db
echo   test-vector    - Test pgvector installation
echo   list-docs      - List recent documents
echo   search "term"  - Search documents by content
echo   vector-status  - Check pgvector and embedding status
echo   run-sql "file" - Execute SQL file
echo.
echo Examples:
echo   psql-shortcuts.bat connect
echo   psql-shortcuts.bat search "contract law"
echo   psql-shortcuts.bat run-sql queries\create-indexes.sql