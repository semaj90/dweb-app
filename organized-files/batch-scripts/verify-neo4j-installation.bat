@echo off
chcp 65001 > nul
echo.
echo ================================================================
echo 🔍 Neo4j Installation Verification
echo ================================================================
echo.

echo 📋 Checking Neo4j installation status...
echo.

REM ==== Check Java ====
echo [1/7] ☕ Checking Java installation...
java -version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Java is installed
    java -version 2>&1 | findstr /C:"version"
) else (
    echo ❌ Java not found - required for Neo4j
)
echo.

REM ==== Check Neo4j Installation ====
echo [2/7] 🔗 Checking Neo4j installation...
if exist "C:\Neo4j" (
    echo ✅ Neo4j directory found: C:\Neo4j
    if exist "C:\Neo4j\bin\neo4j.bat" (
        echo ✅ Neo4j executable found
    ) else (
        echo ❌ Neo4j executable missing
    )
) else (
    echo ❌ Neo4j not installed in expected location (C:\Neo4j)
)
echo.

REM ==== Check PATH ====
echo [3/7] 🛤️  Checking Neo4j in PATH...
where neo4j >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j found in PATH
    where neo4j
) else (
    echo ❌ Neo4j not found in PATH
    echo    Expected: C:\Neo4j\bin should be in PATH
)
echo.

REM ==== Check Neo4j Commands ====
echo [4/7] 🔧 Testing Neo4j commands...
neo4j version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j version command works
    neo4j version
) else (
    echo ❌ Neo4j version command failed
)

neo4j-admin --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j Admin command works
    neo4j-admin --version
) else (
    echo ❌ Neo4j Admin command failed
)
echo.

REM ==== Check Windows Service ====
echo [5/7] 🔧 Checking Neo4j Windows Service...
sc query neo4j >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j service is installed
    sc query neo4j | findstr STATE
) else (
    echo ❌ Neo4j service not installed
)
echo.

REM ==== Check Neo4j Process ====
echo [6/7] 🔄 Checking if Neo4j is running...
tasklist /FI "IMAGENAME eq java.exe" 2>nul | findstr /C:"java.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Java processes found (Neo4j likely running)
    tasklist /FI "IMAGENAME eq java.exe" /FO CSV | findstr /C:"java.exe"
) else (
    echo ⚠️  No Java processes found - Neo4j may not be running
)
echo.

REM ==== Check Neo4j HTTP Interface ====
echo [7/7] 🌐 Testing Neo4j HTTP interface...
curl -s -I http://localhost:7474 | findstr /C:"HTTP" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j HTTP interface responding on port 7474
    curl -s -I http://localhost:7474 | findstr /C:"HTTP"
) else (
    echo ❌ Neo4j HTTP interface not responding on port 7474
)

curl -s -I http://localhost:7687 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Neo4j Bolt interface responding on port 7687
) else (
    echo ⚠️  Neo4j Bolt interface test inconclusive (port 7687)
)

echo.
echo ================================================================
echo 📊 Verification Summary
echo ================================================================
echo.

REM ==== Installation Status ====
if exist "C:\Neo4j\bin\neo4j.bat" (
    echo 🔗 Neo4j Installation: ✅ INSTALLED
) else (
    echo 🔗 Neo4j Installation: ❌ NOT INSTALLED
)

REM ==== PATH Status ====
where neo4j >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🛤️  PATH Configuration: ✅ CONFIGURED
) else (
    echo 🛤️  PATH Configuration: ❌ NOT CONFIGURED
)

REM ==== Service Status ====
sc query neo4j >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🔧 Windows Service: ✅ INSTALLED
) else (
    echo 🔧 Windows Service: ❌ NOT INSTALLED
)

REM ==== Running Status ====
curl -s -I http://localhost:7474 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 🔄 Service Status: ✅ RUNNING
    echo 🌐 Web Interface: http://localhost:7474
    echo 🔌 Bolt Interface: bolt://localhost:7687
    echo 📊 Default Database: legalai
) else (
    echo 🔄 Service Status: ❌ NOT RUNNING
)

echo.
echo 🔑 Default Login (first time):
echo     Username: neo4j
echo     Password: neo4j
echo     (Change password on first login)
echo.

REM ==== Next Steps ====
if not exist "C:\Neo4j\bin\neo4j.bat" (
    echo 📝 Next Steps:
    echo     1. Run install-neo4j-windows.bat as Administrator
    echo     2. Restart command prompt to refresh PATH
    echo     3. Run this verification script again
    echo.
) else (
    curl -s -I http://localhost:7474 >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo 📝 Next Steps to Start Neo4j:
        echo     1. Start as Service: net start neo4j
        echo     2. Or start manually: neo4j console
        echo     3. Open web interface: http://localhost:7474
        echo.
    ) else (
        echo 🎉 Neo4j is fully operational!
        echo     ✅ Open http://localhost:7474 to access Neo4j Browser
        echo     ✅ Ready for Legal AI knowledge graph operations
        echo.
    )
)

echo ================================================================
echo 🔧 Management Commands:
echo     Start:    net start neo4j
echo     Stop:     net stop neo4j  
echo     Console:  neo4j console
echo     Status:   neo4j status
echo     Version:  neo4j version
echo ================================================================
echo.
pause