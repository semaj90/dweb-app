@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo.
echo ================================================================
echo 🔗 Neo4j Database Installation for Legal AI System
echo ================================================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running with Administrator privileges
) else (
    echo ❌ This script requires Administrator privileges
    echo    Please right-click and "Run as Administrator"
    pause
    exit /b 1
)

echo 📋 Starting Neo4j installation process...
echo.

REM ==== STEP 1: Check Java Installation ====
echo [1/8] ☕ Checking Java installation...

java -version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Java is already installed
    java -version
) else (
    echo ⚠️  Java not found - Neo4j requires Java 11 or later
    echo    Installing Java 17...
    
    REM Download and install Java 17
    if not exist "%TEMP%\java-installer.msi" (
        echo    Downloading Java 17 installer...
        powershell -Command "Invoke-WebRequest -Uri 'https://download.oracle.com/java/17/latest/jdk-17_windows-x64_bin.msi' -OutFile '%TEMP%\java-installer.msi'"
    )
    
    echo    Installing Java 17...
    msiexec /i "%TEMP%\java-installer.msi" /quiet /norestart
    
    REM Add Java to PATH
    setx JAVA_HOME "C:\Program Files\Java\jdk-17" /M
    setx PATH "%PATH%;C:\Program Files\Java\jdk-17\bin" /M
    
    echo ✅ Java 17 installed
)

REM ==== STEP 2: Download Neo4j Community Edition ====
echo [2/8] 📥 Downloading Neo4j Community Edition...

set NEO4J_VERSION=5.15.0
set NEO4J_URL=https://neo4j.com/artifact.php?name=neo4j-community-%NEO4J_VERSION%-windows.zip
set NEO4J_ZIP=%TEMP%\neo4j-community-%NEO4J_VERSION%-windows.zip
set NEO4J_EXTRACT_DIR=%TEMP%\neo4j-extract
set NEO4J_INSTALL_DIR=C:\Neo4j

if not exist "%NEO4J_ZIP%" (
    echo    Downloading Neo4j Community %NEO4J_VERSION%...
    powershell -Command "Invoke-WebRequest -Uri '%NEO4J_URL%' -OutFile '%NEO4J_ZIP%'"
    
    if !ERRORLEVEL! NEQ 0 (
        echo ❌ Download failed. Trying alternative URL...
        set NEO4J_URL=https://dist.neo4j.org/neo4j-community-%NEO4J_VERSION%-windows.zip
        powershell -Command "Invoke-WebRequest -Uri '%NEO4J_URL%' -OutFile '%NEO4J_ZIP%'"
    )
) else (
    echo ✅ Neo4j zip already downloaded
)

REM ==== STEP 3: Extract Neo4j ====
echo [3/8] 📂 Extracting Neo4j...

if exist "%NEO4J_EXTRACT_DIR%" rmdir /s /q "%NEO4J_EXTRACT_DIR%"
mkdir "%NEO4J_EXTRACT_DIR%"

powershell -Command "Expand-Archive -Path '%NEO4J_ZIP%' -DestinationPath '%NEO4J_EXTRACT_DIR%' -Force"

if !ERRORLEVEL! EQU 0 (
    echo ✅ Neo4j extracted successfully
) else (
    echo ❌ Failed to extract Neo4j
    pause
    exit /b 1
)

REM ==== STEP 4: Install Neo4j to Program Files ====
echo [4/8] 🚀 Installing Neo4j to %NEO4J_INSTALL_DIR%...

if exist "%NEO4J_INSTALL_DIR%" (
    echo    Removing existing Neo4j installation...
    rmdir /s /q "%NEO4J_INSTALL_DIR%"
)

REM Find extracted folder (it might have different naming)
for /d %%i in ("%NEO4J_EXTRACT_DIR%\neo4j-*") do (
    set EXTRACTED_FOLDER=%%i
    break
)

if exist "%EXTRACTED_FOLDER%" (
    echo    Moving Neo4j to %NEO4J_INSTALL_DIR%...
    move "%EXTRACTED_FOLDER%" "%NEO4J_INSTALL_DIR%"
    echo ✅ Neo4j installed to %NEO4J_INSTALL_DIR%
) else (
    echo ❌ Could not find extracted Neo4j folder
    dir "%NEO4J_EXTRACT_DIR%"
    pause
    exit /b 1
)

REM ==== STEP 5: Add Neo4j to PATH ====
echo [5/8] 🛤️  Adding Neo4j to system PATH...

set NEO4J_BIN=%NEO4J_INSTALL_DIR%\bin

REM Add to system PATH permanently
setx PATH "%PATH%;%NEO4J_BIN%" /M

REM Add to current session PATH
set PATH=%PATH%;%NEO4J_BIN%

echo ✅ Neo4j added to PATH: %NEO4J_BIN%

REM ==== STEP 6: Configure Neo4j ====
echo [6/8] ⚙️  Configuring Neo4j for Legal AI system...

set NEO4J_CONF=%NEO4J_INSTALL_DIR%\conf\neo4j.conf

REM Backup original config
if exist "%NEO4J_CONF%" (
    copy "%NEO4J_CONF%" "%NEO4J_CONF%.backup" >nul
)

REM Create optimized configuration for Legal AI
echo # Neo4j Configuration for Legal AI System > "%NEO4J_CONF%"
echo # Generated on %DATE% %TIME% >> "%NEO4J_CONF%"
echo. >> "%NEO4J_CONF%"
echo # Database location >> "%NEO4J_CONF%"
echo server.default_database=legalai >> "%NEO4J_CONF%"
echo. >> "%NEO4J_CONF%"
echo # Memory settings for legal document processing >> "%NEO4J_CONF%"
echo server.memory.heap.initial_size=1G >> "%NEO4J_CONF%"
echo server.memory.heap.max_size=2G >> "%NEO4J_CONF%"
echo server.memory.pagecache.size=1G >> "%NEO4J_CONF%"
echo. >> "%NEO4J_CONF%"
echo # Network settings >> "%NEO4J_CONF%"
echo server.default_listen_address=0.0.0.0 >> "%NEO4J_CONF%"
echo server.bolt.listen_address=:7687 >> "%NEO4J_CONF%"
echo server.http.listen_address=:7474 >> "%NEO4J_CONF%"
echo. >> "%NEO4J_CONF%"
echo # Security settings >> "%NEO4J_CONF%"
echo server.directories.logs=%NEO4J_INSTALL_DIR%\logs >> "%NEO4J_CONF%"
echo server.directories.data=%NEO4J_INSTALL_DIR%\data >> "%NEO4J_CONF%"
echo. >> "%NEO4J_CONF%"
echo # Performance tuning for legal data >> "%NEO4J_CONF%"
echo db.tx_log.rotation.retention_policy=1G size >> "%NEO4J_CONF%"
echo server.jvm.additional=-XX:+UseG1GC >> "%NEO4J_CONF%"
echo server.jvm.additional=-XX:+UnlockExperimentalVMOptions >> "%NEO4J_CONF%"

echo ✅ Neo4j configured for Legal AI system

REM ==== STEP 7: Install Neo4j as Windows Service ====
echo [7/8] 🔧 Installing Neo4j as Windows Service...

cd /d "%NEO4J_INSTALL_DIR%\bin"

REM Install service
neo4j-admin.bat service install

if !ERRORLEVEL! EQU 0 (
    echo ✅ Neo4j service installed successfully
    
    REM Start the service
    echo    Starting Neo4j service...
    net start neo4j
    
    if !ERRORLEVEL! EQU 0 (
        echo ✅ Neo4j service started successfully
    ) else (
        echo ⚠️  Neo4j service installation completed but failed to start
        echo    You can start it manually with: net start neo4j
    )
) else (
    echo ⚠️  Neo4j service installation failed
    echo    You can run Neo4j manually with: neo4j console
)

REM ==== STEP 8: Verify Installation ====
echo [8/8] ✅ Verifying Neo4j installation...

echo    Checking Neo4j command...
neo4j version
if !ERRORLEVEL! EQU 0 (
    echo ✅ Neo4j command accessible
) else (
    echo ❌ Neo4j command not found in PATH
)

echo    Checking Neo4j Admin command...
neo4j-admin --version
if !ERRORLEVEL! EQU 0 (
    echo ✅ Neo4j Admin command accessible
) else (
    echo ❌ Neo4j Admin command not found
)

echo    Waiting for Neo4j to fully start...
timeout /t 10 /nobreak >nul

echo    Testing Neo4j HTTP endpoint...
curl -s http://localhost:7474 >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo ✅ Neo4j HTTP interface accessible at http://localhost:7474
) else (
    echo ⚠️  Neo4j HTTP interface not yet accessible
)

REM ==== CLEANUP ====
echo 🧹 Cleaning up temporary files...
if exist "%NEO4J_ZIP%" del "%NEO4J_ZIP%"
if exist "%NEO4J_EXTRACT_DIR%" rmdir /s /q "%NEO4J_EXTRACT_DIR%"
if exist "%TEMP%\java-installer.msi" del "%TEMP%\java-installer.msi"

echo.
echo ================================================================
echo 🎉 Neo4j Installation Complete!
echo ================================================================
echo.
echo 📊 Installation Summary:
echo     Neo4j Version: Community %NEO4J_VERSION%
echo     Installation Path: %NEO4J_INSTALL_DIR%
echo     Service Name: neo4j
echo     HTTP Interface: http://localhost:7474
echo     Bolt Interface: bolt://localhost:7687
echo     Database: legalai
echo.
echo 🔑 Default Credentials (first login):
echo     Username: neo4j
echo     Password: neo4j
echo     (You will be prompted to change on first login)
echo.
echo 🌐 Web Interface:
echo     Open: http://localhost:7474
echo     Login with neo4j/neo4j and set new password
echo.
echo 🔧 Management Commands:
echo     Start Service:    net start neo4j
echo     Stop Service:     net stop neo4j
echo     Console Mode:     neo4j console
echo     Status Check:     neo4j status
echo.
echo 🔗 Integration with Legal AI:
echo     The database 'legalai' is configured for your system.
echo     Memory optimized for legal document processing.
echo     Ready for knowledge graph operations.
echo.

REM Create quick access scripts
echo Creating management scripts...
echo @echo off > "%NEO4J_INSTALL_DIR%\start-neo4j.bat"
echo echo Starting Neo4j service... >> "%NEO4J_INSTALL_DIR%\start-neo4j.bat"
echo net start neo4j >> "%NEO4J_INSTALL_DIR%\start-neo4j.bat"
echo echo Neo4j started. Web interface: http://localhost:7474 >> "%NEO4J_INSTALL_DIR%\start-neo4j.bat"
echo pause >> "%NEO4J_INSTALL_DIR%\start-neo4j.bat"

echo @echo off > "%NEO4J_INSTALL_DIR%\stop-neo4j.bat"
echo echo Stopping Neo4j service... >> "%NEO4J_INSTALL_DIR%\stop-neo4j.bat"
echo net stop neo4j >> "%NEO4J_INSTALL_DIR%\stop-neo4j.bat"
echo echo Neo4j stopped. >> "%NEO4J_INSTALL_DIR%\stop-neo4j.bat"
echo pause >> "%NEO4J_INSTALL_DIR%\stop-neo4j.bat"

echo ✅ Management scripts created in %NEO4J_INSTALL_DIR%
echo.
echo Press any key to open Neo4j Browser...
pause >nul
start http://localhost:7474

endlocal