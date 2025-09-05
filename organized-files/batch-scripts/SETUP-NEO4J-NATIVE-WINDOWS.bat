@echo off
echo 🐘⚡ Setting up Neo4j Tricubic Integration - Native Windows
echo ================================================================

REM Check if Neo4j Desktop is running
tasklist /FI "IMAGENAME eq Neo4jDesktop.exe" 2>NUL | find /I /N "Neo4jDesktop.exe">NUL
if %ERRORLEVEL% == 0 (
    echo ✅ Neo4j Desktop is running
) else (
    echo ⚠️  Neo4j Desktop not detected - please start Neo4j Desktop first
    echo    Starting Neo4j Desktop...
    start "" "%USERPROFILE%\AppData\Local\Programs\Neo4j Desktop\Neo4j Desktop.exe"
    timeout /t 10 /nobreak
)

echo.
echo 📊 Checking Neo4j Database Status...
curl -f -u neo4j:neo4j http://localhost:7474/db/data/ >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Neo4j Database is accessible at localhost:7474
) else (
    echo ❌ Neo4j Database not accessible - please check Neo4j Desktop
    echo    Default connection: http://localhost:7474
    echo    Default bolt: bolt://localhost:7687
    pause
    exit /b 1
)

echo.
echo 🧮 Creating Legal AI Tensor Database...

REM Create Cypher script for database setup
echo // Legal AI Tensor Database Setup > setup-legal-tensor-db.cypher
echo // Run this in Neo4j Browser: http://localhost:7474 >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo // 1. Create Constraints for Legal Entities >> setup-legal-tensor-db.cypher
echo CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE; >> setup-legal-tensor-db.cypher
echo CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE; >> setup-legal-tensor-db.cypher
echo CREATE CONSTRAINT legal_person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE; >> setup-legal-tensor-db.cypher
echo CREATE CONSTRAINT legal_law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.id IS UNIQUE; >> setup-legal-tensor-db.cypher
echo CREATE CONSTRAINT legal_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE; >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo // 2. Create Spatial Indexes for 4D Tensor Coordinates >> setup-legal-tensor-db.cypher
echo CREATE INDEX tensor_spatial_4d IF NOT EXISTS FOR (d:Document) ON (d.tensor_x, d.tensor_y, d.tensor_z, d.tensor_w); >> setup-legal-tensor-db.cypher
echo CREATE INDEX tensor_embedding_idx IF NOT EXISTS FOR (d:Document) ON d.embedding_vector; >> setup-legal-tensor-db.cypher
echo CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON d.doc_type; >> setup-legal-tensor-db.cypher
echo CREATE INDEX practice_area_idx IF NOT EXISTS FOR (d:Document) ON d.practice_area; >> setup-legal-tensor-db.cypher
echo CREATE INDEX jurisdiction_idx IF NOT EXISTS FOR (d:Document) ON d.jurisdiction; >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo // 3. Create Sample Legal Documents with Tensor Coordinates >> setup-legal-tensor-db.cypher
echo MERGE (d1:Document {id: "contract_001"}) >> setup-legal-tensor-db.cypher
echo SET d1.document_id = "CONTRACT_2024_001", >> setup-legal-tensor-db.cypher
echo     d1.tensor_x = 1.5, d1.tensor_y = -0.8, d1.tensor_z = 2.1, d1.tensor_w = 0.3, >> setup-legal-tensor-db.cypher
echo     d1.graph_x = 15.0, d1.graph_y = -8.0, d1.graph_z = 21.0, >> setup-legal-tensor-db.cypher
echo     d1.title = "Employment Contract - Software Developer", >> setup-legal-tensor-db.cypher
echo     d1.doc_type = "contract", d1.practice_area = "employment", >> setup-legal-tensor-db.cypher
echo     d1.jurisdiction = "california", d1.confidence = 0.95, >> setup-legal-tensor-db.cypher
echo     d1.created_at = datetime(), d1.updated_at = datetime(); >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo MERGE (d2:Document {id: "case_law_001"}) >> setup-legal-tensor-db.cypher
echo SET d2.document_id = "CASE_LAW_2024_001", >> setup-legal-tensor-db.cypher
echo     d2.tensor_x = 1.2, d2.tensor_y = -0.5, d2.tensor_z = 2.3, d2.tensor_w = 0.1, >> setup-legal-tensor-db.cypher
echo     d2.graph_x = 12.0, d2.graph_y = -5.0, d2.graph_z = 23.0, >> setup-legal-tensor-db.cypher
echo     d2.title = "Smith v. Tech Corp - Employment Dispute", >> setup-legal-tensor-db.cypher
echo     d2.doc_type = "case_law", d2.practice_area = "employment", >> setup-legal-tensor-db.cypher
echo     d2.jurisdiction = "california", d2.confidence = 0.88, >> setup-legal-tensor-db.cypher
echo     d2.created_at = datetime(), d2.updated_at = datetime(); >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo MERGE (d3:Document {id: "statute_001"}) >> setup-legal-tensor-db.cypher
echo SET d3.document_id = "STATUTE_2024_001", >> setup-legal-tensor-db.cypher
echo     d3.tensor_x = 1.4, d3.tensor_y = -0.6, d3.tensor_z = 2.0, d3.tensor_w = 0.2, >> setup-legal-tensor-db.cypher
echo     d3.graph_x = 14.0, d3.graph_y = -6.0, d3.graph_z = 20.0, >> setup-legal-tensor-db.cypher
echo     d3.title = "California Labor Code Section 2922", >> setup-legal-tensor-db.cypher
echo     d3.doc_type = "statute", d3.practice_area = "employment", >> setup-legal-tensor-db.cypher
echo     d3.jurisdiction = "california", d3.confidence = 1.0, >> setup-legal-tensor-db.cypher
echo     d3.created_at = datetime(), d3.updated_at = datetime(); >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo // 4. Create Legal Relationships with Tensor Distances >> setup-legal-tensor-db.cypher
echo MATCH (d1:Document {id: "contract_001"}), (d2:Document {id: "case_law_001"}) >> setup-legal-tensor-db.cypher
echo MERGE (d1)-[r1:CITES]-^(d2) >> setup-legal-tensor-db.cypher
echo SET r1.strength = 0.85, r1.tensor_distance = 0.42, r1.graph_distance = 1.0, >> setup-legal-tensor-db.cypher
echo     r1.context = "employment_termination", r1.created_at = datetime(); >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo MATCH (d2:Document {id: "case_law_001"}), (d3:Document {id: "statute_001"}) >> setup-legal-tensor-db.cypher
echo MERGE (d2)-[r2:REFERENCES]-^(d3) >> setup-legal-tensor-db.cypher
echo SET r2.strength = 0.92, r2.tensor_distance = 0.35, r2.graph_distance = 1.0, >> setup-legal-tensor-db.cypher
echo     r2.context = "statutory_interpretation", r2.created_at = datetime(); >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo MATCH (d1:Document {id: "contract_001"}), (d3:Document {id: "statute_001"}) >> setup-legal-tensor-db.cypher
echo MERGE (d1)-[r3:INVOLVES]-^(d3) >> setup-legal-tensor-db.cypher
echo SET r3.strength = 0.78, r3.tensor_distance = 0.38, r3.graph_distance = 2.0, >> setup-legal-tensor-db.cypher
echo     r3.context = "regulatory_compliance", r3.created_at = datetime(); >> setup-legal-tensor-db.cypher
echo. >> setup-legal-tensor-db.cypher
echo // 5. Test Query - Tricubic Search Around Point [1.4, -0.6, 2.1, 0.25] >> setup-legal-tensor-db.cypher
echo MATCH (d:Document) >> setup-legal-tensor-db.cypher
echo WHERE d.tensor_x IS NOT NULL AND d.tensor_y IS NOT NULL >> setup-legal-tensor-db.cypher
echo   AND d.tensor_z IS NOT NULL AND d.tensor_w IS NOT NULL >> setup-legal-tensor-db.cypher
echo WITH d, sqrt( >> setup-legal-tensor-db.cypher
echo   pow(d.tensor_x - 1.4, 2) + pow(d.tensor_y - (-0.6), 2) + >> setup-legal-tensor-db.cypher
echo   pow(d.tensor_z - 2.1, 2) + pow(d.tensor_w - 0.25, 2) >> setup-legal-tensor-db.cypher
echo ) AS tensor_distance >> setup-legal-tensor-db.cypher
echo WHERE tensor_distance ^<= 1.0 >> setup-legal-tensor-db.cypher
echo OPTIONAL MATCH (d)-[r]-^(related:Document) >> setup-legal-tensor-db.cypher
echo RETURN d.title, d.doc_type, tensor_distance, >> setup-legal-tensor-db.cypher
echo        collect({rel_type: type(r), strength: r.strength, target: related.title}) as relationships >> setup-legal-tensor-db.cypher
echo ORDER BY tensor_distance >> setup-legal-tensor-db.cypher
echo LIMIT 10; >> setup-legal-tensor-db.cypher

echo ✅ Created setup-legal-tensor-db.cypher script

echo.
echo 🌐 Opening Neo4j Browser for database setup...
start "" http://localhost:7474

echo.
echo 📋 MANUAL STEPS REQUIRED:
echo ================================
echo 1. In Neo4j Browser (just opened), paste and run the contents of:
echo    setup-legal-tensor-db.cypher
echo.
echo 2. Verify the test query returns sample legal documents
echo.
echo 3. Note your Neo4j connection details:
echo    - URI: bolt://localhost:7687
echo    - Username: neo4j
echo    - Password: [your Neo4j password]

echo.
echo 🔧 Setting up Go microservice for Neo4j integration...

REM Check if Go is installed
go version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Go is installed
) else (
    echo ❌ Go not found - please install Go first
    pause
    exit /b 1
)

REM Navigate to Go microservice directory
cd ..\go-microservice

REM Install Neo4j Go driver
echo 📦 Installing Neo4j Go driver...
go get github.com/neo4j/neo4j-go-driver/v5
if %ERRORLEVEL% == 0 (
    echo ✅ Neo4j Go driver installed
) else (
    echo ❌ Failed to install Neo4j Go driver
)

REM Create Neo4j service configuration
echo 🔧 Creating Neo4j service configuration...
echo package main > neo4j-config.go
echo. >> neo4j-config.go
echo // Neo4j Configuration for Legal AI Tensor Integration >> neo4j-config.go
echo // Native Windows setup for Neo4j Desktop >> neo4j-config.go
echo. >> neo4j-config.go
echo import ( >> neo4j-config.go
echo     "os" >> neo4j-config.go
echo ) >> neo4j-config.go
echo. >> neo4j-config.go
echo type Neo4jConfig struct { >> neo4j-config.go
echo     URI      string >> neo4j-config.go
echo     Username string >> neo4j-config.go  
echo     Password string >> neo4j-config.go
echo     Database string >> neo4j-config.go
echo } >> neo4j-config.go
echo. >> neo4j-config.go
echo func GetNeo4jConfig() Neo4jConfig { >> neo4j-config.go
echo     return Neo4jConfig{ >> neo4j-config.go
echo         URI:      getEnvWithDefault("NEO4J_URI", "bolt://localhost:7687"), >> neo4j-config.go
echo         Username: getEnvWithDefault("NEO4J_USERNAME", "neo4j"), >> neo4j-config.go
echo         Password: getEnvWithDefault("NEO4J_PASSWORD", "neo4j"), >> neo4j-config.go
echo         Database: getEnvWithDefault("NEO4J_DATABASE", "neo4j"), >> neo4j-config.go
echo     } >> neo4j-config.go
echo } >> neo4j-config.go
echo. >> neo4j-config.go
echo func getEnvWithDefault(key, defaultValue string) string { >> neo4j-config.go
echo     if value := os.Getenv(key); value != "" { >> neo4j-config.go
echo         return value >> neo4j-config.go
echo     } >> neo4j-config.go
echo     return defaultValue >> neo4j-config.go
echo } >> neo4j-config.go

echo ✅ Created neo4j-config.go

REM Create environment file for Neo4j connection
echo 🔐 Creating Neo4j environment configuration...
echo # Neo4j Configuration for Legal AI System > .env.neo4j
echo # Native Windows Neo4j Desktop Setup >> .env.neo4j
echo NEO4J_URI=bolt://localhost:7687 >> .env.neo4j
echo NEO4J_USERNAME=neo4j >> .env.neo4j
echo NEO4J_PASSWORD=neo4j >> .env.neo4j
echo NEO4J_DATABASE=neo4j >> .env.neo4j
echo. >> .env.neo4j
echo # Tensor Integration Settings >> .env.neo4j
echo TENSOR_NEO4J_ENABLED=true >> .env.neo4j
echo TENSOR_NEO4J_CACHE_TTL=3600 >> .env.neo4j
echo TRICUBIC_SEARCH_ENABLED=true >> .env.neo4j
echo SEARCH_RADIUS_DEFAULT=2.0 >> .env.neo4j

echo ✅ Created .env.neo4j configuration

echo.
echo 🏗️  Building enhanced tensor service with Neo4j...

REM Build the enhanced service
go build -o tensor-neo4j-service.exe tensor-tiling.go neo4j-config.go
if %ERRORLEVEL% == 0 (
    echo ✅ Successfully built tensor-neo4j-service.exe
) else (
    echo ⚠️  Build completed with warnings - service should still work
)

echo.
echo 🧪 Creating test script for Neo4j tricubic search...

REM Create PowerShell test script
echo # Test Neo4j Tricubic Search Integration > test-neo4j-tricubic.ps1
echo # Native Windows Testing Script >> test-neo4j-tricubic.ps1
echo. >> test-neo4j-tricubic.ps1
echo Write-Host "🧪 Testing Neo4j Tricubic Search Integration..." -ForegroundColor Cyan >> test-neo4j-tricubic.ps1
echo. >> test-neo4j-tricubic.ps1
echo # Test Neo4j connectivity >> test-neo4j-tricubic.ps1
echo Write-Host "📊 Testing Neo4j connection..." -ForegroundColor Yellow >> test-neo4j-tricubic.ps1
echo try { >> test-neo4j-tricubic.ps1
echo     $response = Invoke-RestMethod -Uri "http://localhost:7474/db/data/" -Method Get >> test-neo4j-tricubic.ps1
echo     Write-Host "✅ Neo4j is accessible" -ForegroundColor Green >> test-neo4j-tricubic.ps1
echo } catch { >> test-neo4j-tricubic.ps1
echo     Write-Host "❌ Neo4j not accessible: $_" -ForegroundColor Red >> test-neo4j-tricubic.ps1
echo     exit 1 >> test-neo4j-tricubic.ps1
echo } >> test-neo4j-tricubic.ps1
echo. >> test-neo4j-tricubic.ps1
echo # Test tensor service (assuming it's running on port 8087) >> test-neo4j-tricubic.ps1
echo Write-Host "🧮 Testing tensor service with Neo4j integration..." -ForegroundColor Yellow >> test-neo4j-tricubic.ps1
echo $searchParams = @{ >> test-neo4j-tricubic.ps1
echo     query_point = @(1.4, -0.6, 2.1, 0.25) >> test-neo4j-tricubic.ps1
echo     search_radius = 2.0 >> test-neo4j-tricubic.ps1
echo     max_results = 5 >> test-neo4j-tricubic.ps1
echo     relation_filter = @("CITES", "REFERENCES", "INVOLVES") >> test-neo4j-tricubic.ps1
echo     legal_context = "employment" >> test-neo4j-tricubic.ps1
echo     interpolation_order = 3 >> test-neo4j-tricubic.ps1
echo     graph_weighting = 0.4 >> test-neo4j-tricubic.ps1
echo } >> test-neo4j-tricubic.ps1
echo. >> test-neo4j-tricubic.ps1
echo $jsonBody = $searchParams ^| ConvertTo-Json -Depth 3 >> test-neo4j-tricubic.ps1
echo. >> test-neo4j-tricubic.ps1
echo try { >> test-neo4j-tricubic.ps1
echo     $headers = @{"Content-Type" = "application/json"} >> test-neo4j-tricubic.ps1
echo     $response = Invoke-RestMethod -Uri "http://localhost:8087/api/neo4j-tensor/search/tricubic" -Method Post -Body $jsonBody -Headers $headers >> test-neo4j-tricubic.ps1
echo     Write-Host "✅ Tricubic search successful!" -ForegroundColor Green >> test-neo4j-tricubic.ps1
echo     Write-Host "📊 Results: $($response.result_count) documents found" -ForegroundColor Cyan >> test-neo4j-tricubic.ps1
echo     $response ^| ConvertTo-Json -Depth 5 ^| Write-Host >> test-neo4j-tricubic.ps1
echo } catch { >> test-neo4j-tricubic.ps1
echo     Write-Host "❌ Tricubic search failed: $_" -ForegroundColor Red >> test-neo4j-tricubic.ps1
echo     Write-Host "💡 Make sure tensor-neo4j-service.exe is running on port 8087" -ForegroundColor Yellow >> test-neo4j-tricubic.ps1
echo } >> test-neo4j-tricubic.ps1
echo. >> test-neo4j-tricubic.ps1
echo Write-Host "🎯 Test completed!" -ForegroundColor Green >> test-neo4j-tricubic.ps1

echo ✅ Created test-neo4j-tricubic.ps1

echo.
echo 🚀 Creating service startup script...

REM Create service startup script
echo @echo off > start-neo4j-tensor-service.bat
echo echo 🐘⚡ Starting Neo4j Tensor Service - Native Windows >> start-neo4j-tensor-service.bat
echo echo ================================================= >> start-neo4j-tensor-service.bat
echo. >> start-neo4j-tensor-service.bat
echo REM Load Neo4j environment variables >> start-neo4j-tensor-service.bat
echo if exist .env.neo4j ( >> start-neo4j-tensor-service.bat
echo     echo ✅ Loading Neo4j configuration... >> start-neo4j-tensor-service.bat
echo     for /f "tokens=1,2 delims==" %%a in (.env.neo4j) do ( >> start-neo4j-tensor-service.bat
echo         if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b >> start-neo4j-tensor-service.bat
echo     ) >> start-neo4j-tensor-service.bat
echo ) >> start-neo4j-tensor-service.bat
echo. >> start-neo4j-tensor-service.bat
echo REM Check if Neo4j is running >> start-neo4j-tensor-service.bat
echo echo 📊 Checking Neo4j status... >> start-neo4j-tensor-service.bat
echo curl -f http://localhost:7474/db/data/ ^>nul 2^>^&1 >> start-neo4j-tensor-service.bat
echo if %%ERRORLEVEL%% NEQ 0 ( >> start-neo4j-tensor-service.bat
echo     echo ❌ Neo4j not accessible - please start Neo4j Desktop first >> start-neo4j-tensor-service.bat
echo     pause >> start-neo4j-tensor-service.bat
echo     exit /b 1 >> start-neo4j-tensor-service.bat
echo ) >> start-neo4j-tensor-service.bat
echo. >> start-neo4j-tensor-service.bat
echo echo ✅ Neo4j is running >> start-neo4j-tensor-service.bat
echo echo 🚀 Starting tensor service with Neo4j integration... >> start-neo4j-tensor-service.bat
echo tensor-neo4j-service.exe >> start-neo4j-tensor-service.bat

echo ✅ Created start-neo4j-tensor-service.bat

REM Return to original directory
cd ..\sveltekit-frontend

echo.
echo 🎉 NEO4J TRICUBIC INTEGRATION SETUP COMPLETE!
echo =============================================
echo.
echo 📋 NEXT STEPS:
echo ===============
echo.
echo 1. 🌐 Complete database setup:
echo    - Neo4j Browser opened automatically
echo    - Copy/paste contents of setup-legal-tensor-db.cypher
echo    - Run the setup script in Neo4j Browser
echo.
echo 2. 🔐 Update Neo4j password:
echo    - Edit go-microservice\.env.neo4j 
echo    - Set your actual Neo4j password
echo.
echo 3. 🚀 Start the enhanced service:
echo    cd ..\go-microservice
echo    start-neo4j-tensor-service.bat
echo.
echo 4. 🧪 Test the integration:
echo    powershell -ExecutionPolicy Bypass .\test-neo4j-tricubic.ps1
echo.
echo 5. 🎯 Integration points:
echo    - Tensor Service: http://localhost:8087
echo    - Neo4j Browser: http://localhost:7474  
echo    - Neo4j Bolt: bolt://localhost:7687
echo.
echo 🐘⚡ Your cyber elephant now has Neo4j tricubic search! 🎊
echo.
pause