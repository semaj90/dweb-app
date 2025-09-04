@echo off
echo ===========================================
echo    pgAdmin Connection Setup for Legal AI
echo ===========================================
echo.
echo Your PostgreSQL database is ready for pgAdmin connection!
echo.
echo CONNECTION DETAILS:
echo -------------------
echo Host:     localhost
echo Port:     5432
echo Database: legal_ai_db
echo Username: legal_admin
echo Password: 123456
echo.
echo TABLES AVAILABLE (21 total):
echo ----------------------------
echo - users, sessions, user_profiles
echo - cases, evidence, case_scores
echo - legal_documents, document_chunks
echo - reports, legal_analysis_sessions
echo - persons_of_interest, criminals
echo - rag_sessions, rag_messages
echo - vector_metadata, embedding_cache
echo - auto_tags, canvas_states
echo - statutes, user_ai_queries, keys
echo.
echo PGVECTOR EXTENSION: v0.8.0 (INSTALLED)
echo.
echo ===========================================
echo.
echo To connect pgAdmin:
echo 1. Right-click "Servers" in pgAdmin
echo 2. Create ^> Server...
echo 3. General Tab - Name: "Legal AI Database"
echo 4. Connection Tab - Use details above
echo 5. Save password: CHECK THIS BOX
echo 6. Click Save
echo.
echo Starting pgAdmin (if installed)...
echo.
start "" "C:\Program Files\pgAdmin 4\bin\pgAdmin4.exe" 2>nul || (
    start "" "C:\Program Files (x86)\pgAdmin 4\bin\pgAdmin4.exe" 2>nul || (
        echo pgAdmin not found in default locations.
        echo Please launch pgAdmin manually and use connection details above.
        echo.
        echo Common pgAdmin locations:
        echo - C:\Program Files\pgAdmin 4\bin\pgAdmin4.exe
        echo - C:\Program Files (x86)\pgAdmin 4\bin\pgAdmin4.exe
        echo - Or search for "pgAdmin" in Start Menu
    )
)
echo.
pause