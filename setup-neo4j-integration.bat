@echo off
chcp 65001 > nul
echo.
echo ================================================================
echo 🔗 Neo4j Integration Setup for Legal AI System
echo ================================================================
echo.

echo 📋 Setting up Neo4j integration...

REM ==== STEP 1: Check Neo4j Installation ====
echo [1/6] 🔍 Checking Neo4j installation...

where neo4j >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j found in PATH
    neo4j version
) else (
    echo ❌ Neo4j not found in PATH
    echo    Please run install-neo4j-windows.bat first
    pause
    exit /b 1
)

REM ==== STEP 2: Check Neo4j Service ====
echo [2/6] 🔧 Checking Neo4j service status...

sc query neo4j >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j service is installed
    sc query neo4j | findstr STATE
    
    REM Try to start if stopped
    sc query neo4j | findstr "RUNNING" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo    Starting Neo4j service...
        net start neo4j
    )
) else (
    echo ⚠️  Neo4j service not installed - installing now...
    cd /d "C:\Neo4j\bin" 2>nul || (
        echo ❌ Neo4j not found in C:\Neo4j\bin
        echo    Please run install-neo4j-windows.bat first
        pause
        exit /b 1
    )
    neo4j-admin service install
    net start neo4j
)

REM ==== STEP 3: Test Neo4j Connection ====
echo [3/6] 🌐 Testing Neo4j connection...

timeout /t 5 /nobreak >nul
curl -s http://localhost:7474 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j HTTP interface accessible at http://localhost:7474
) else (
    echo ❌ Neo4j HTTP interface not accessible
    echo    Checking if Neo4j is starting...
    timeout /t 10 /nobreak >nul
    curl -s http://localhost:7474 >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Neo4j now accessible (startup delay)
    ) else (
        echo ❌ Neo4j still not accessible - check logs
    )
)

REM ==== STEP 4: Set Environment Variables ====
echo [4/6] 🔧 Setting up environment variables...

set NEO4J_URI=neo4j://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=legalai123
set NEO4J_DATABASE=legalai

REM Set permanent environment variables
setx NEO4J_URI "%NEO4J_URI%"
setx NEO4J_USER "%NEO4J_USER%"
setx NEO4J_PASSWORD "%NEO4J_PASSWORD%"
setx NEO4J_DATABASE "%NEO4J_DATABASE%"

echo ✅ Environment variables set:
echo    NEO4J_URI: %NEO4J_URI%
echo    NEO4J_USER: %NEO4J_USER%
echo    NEO4J_PASSWORD: %NEO4J_PASSWORD%
echo    NEO4J_DATABASE: %NEO4J_DATABASE%

REM ==== STEP 5: Create Legal AI Database Schema ====
echo [5/6] 📊 Creating Legal AI knowledge graph schema...

REM Create Cypher script for legal AI schema
echo // Legal AI Knowledge Graph Schema > legal-ai-schema.cypher
echo // Case Management Nodes and Relationships >> legal-ai-schema.cypher
echo. >> legal-ai-schema.cypher
echo // Create constraints and indexes >> legal-ai-schema.cypher
echo CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE; >> legal-ai-schema.cypher
echo CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE; >> legal-ai-schema.cypher
echo CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE; >> legal-ai-schema.cypher
echo CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE; >> legal-ai-schema.cypher
echo. >> legal-ai-schema.cypher
echo // Create indexes for performance >> legal-ai-schema.cypher
echo CREATE INDEX case_status_idx IF NOT EXISTS FOR (c:Case) ON (c.status); >> legal-ai-schema.cypher
echo CREATE INDEX evidence_type_idx IF NOT EXISTS FOR (e:Evidence) ON (e.type); >> legal-ai-schema.cypher
echo CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name); >> legal-ai-schema.cypher
echo CREATE INDEX document_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title); >> legal-ai-schema.cypher
echo. >> legal-ai-schema.cypher
echo // Sample data for testing >> legal-ai-schema.cypher
echo CREATE (c1:Case {id: 'CASE-2024-001', title: 'Sample Legal Case', status: 'active', created: datetime()}); >> legal-ai-schema.cypher
echo CREATE (p1:Person {id: 'PERSON-001', name: 'John Doe', role: 'defendant'}); >> legal-ai-schema.cypher
echo CREATE (e1:Evidence {id: 'EVIDENCE-001', type: 'document', title: 'Contract Agreement', created: datetime()}); >> legal-ai-schema.cypher
echo CREATE (d1:Document {id: 'DOC-001', title: 'Legal Brief', content: 'Sample legal document content'}); >> legal-ai-schema.cypher
echo. >> legal-ai-schema.cypher
echo // Create relationships >> legal-ai-schema.cypher
echo MATCH (c:Case {id: 'CASE-2024-001'}), (p:Person {id: 'PERSON-001'}) >> legal-ai-schema.cypher
echo CREATE (c)-[:INVOLVES]->(p); >> legal-ai-schema.cypher
echo MATCH (c:Case {id: 'CASE-2024-001'}), (e:Evidence {id: 'EVIDENCE-001'}) >> legal-ai-schema.cypher
echo CREATE (c)-[:HAS_EVIDENCE]->(e); >> legal-ai-schema.cypher
echo MATCH (e:Evidence {id: 'EVIDENCE-001'}), (d:Document {id: 'DOC-001'}) >> legal-ai-schema.cypher
echo CREATE (e)-[:CONTAINS]->(d); >> legal-ai-schema.cypher

echo ✅ Legal AI schema created in legal-ai-schema.cypher

REM ==== STEP 6: Test Go Server Integration ====
echo [6/6] 🧪 Testing Go server Neo4j integration...

REM Check if Go server is running
curl -s http://localhost:8081/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Go server is running - testing Neo4j integration...
    
    REM Test Neo4j status endpoint
    curl -s http://localhost:8081/neo4j-status
    echo.
    echo ✅ Neo4j integration test complete
) else (
    echo ⚠️  Go server not running on port 8081
    echo    Start the Go server to test Neo4j integration
)

echo.
echo ================================================================
echo 🎉 Neo4j Integration Setup Complete!
echo ================================================================
echo.
echo 📊 Connection Details:
echo     Neo4j URI: %NEO4J_URI%
echo     User: %NEO4J_USER%
echo     Password: %NEO4J_PASSWORD%
echo     Database: %NEO4J_DATABASE%
echo.
echo 🌐 Access Points:
echo     Neo4j Browser: http://localhost:7474
echo     Go Server Health: http://localhost:8081/health
echo     Neo4j Status: http://localhost:8081/neo4j-status
echo     Graph Query: POST http://localhost:8081/graph-query
echo.
echo 🔧 Management Commands:
echo     Start Neo4j: net start neo4j
echo     Stop Neo4j: net stop neo4j
echo     Neo4j Console: neo4j console
echo     View Logs: type "C:\Neo4j\logs\neo4j.log"
echo.
echo 📝 Next Steps:
echo     1. Open http://localhost:7474 in browser
echo     2. Login with neo4j/%NEO4J_PASSWORD%
echo     3. Run: LOAD CSV FROM 'file:///legal-ai-schema.cypher' AS line RETURN line
echo     4. Test with: curl -X POST http://localhost:8081/graph-query
echo.
pause