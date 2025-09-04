@echo off
REM Neo4j Service Controller
REM Provides easy Neo4j management

echo ======================================
echo        Neo4j Service Controller
echo ======================================
echo.
echo   1. Start Neo4j
echo   2. Stop Neo4j  
echo   3. Restart Neo4j
echo   4. Open Neo4j Browser
echo   5. Check Status
echo   6. Exit
echo.

set /p choice="Select option (1-6): "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto browser
if "%choice%"=="5" goto status
if "%choice%"=="6" goto end

:start
echo.
echo Starting Neo4j...
if exist "neo4j-community-5.23.0\bin\neo4j.bat" (
    call neo4j-community-5.23.0\bin\neo4j.bat console
) else if exist "neo4j-community-5.21.2\bin\neo4j.bat" (
    call neo4j-community-5.21.2\bin\neo4j.bat console
) else (
    echo Neo4j not found!
    echo Please download from: https://neo4j.com/download-center/#community
)
goto end

:stop
echo.
echo Stopping Neo4j...
if exist "neo4j-community-5.23.0\bin\neo4j.bat" (
    call neo4j-community-5.23.0\bin\neo4j.bat stop
) else if exist "neo4j-community-5.21.2\bin\neo4j.bat" (
    call neo4j-community-5.21.2\bin\neo4j.bat stop
)
echo Neo4j stopped.
pause
goto end

:restart
echo.
echo Restarting Neo4j...
if exist "neo4j-community-5.23.0\bin\neo4j.bat" (
    call neo4j-community-5.23.0\bin\neo4j.bat stop
    timeout /t 3 /nobreak >nul
    call neo4j-community-5.23.0\bin\neo4j.bat console
) else if exist "neo4j-community-5.21.2\bin\neo4j.bat" (
    call neo4j-community-5.21.2\bin\neo4j.bat stop
    timeout /t 3 /nobreak >nul
    call neo4j-community-5.21.2\bin\neo4j.bat console
)
goto end

:browser
echo.
echo Opening Neo4j Browser...
start http://localhost:7474
echo Browser opened. Default credentials: neo4j/password
pause
goto end

:status
echo.
echo Checking Neo4j status...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 7474" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Neo4j is RUNNING on port 7474
) else (
    echo Neo4j is NOT RUNNING
)
pause
goto end

:end
exit /b