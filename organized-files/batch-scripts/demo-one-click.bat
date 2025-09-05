@echo off
echo 🎯 Evidence Processing System - One-Click Demo
echo ==============================================

echo.
echo This demo will:
echo 1. 🔍 Detect your existing services (PostgreSQL with password 123456)
echo 2. 📦 Install missing portable services
echo 3. 🚀 Start everything automatically
echo 4. 🧪 Run a complete system test
echo 5. 📊 Show you the web interfaces
echo.
echo ⏱️ Estimated time: 3-5 minutes
echo.
pause

echo.
echo 🔍 Step 1/5: Smart Service Detection
echo ===================================
cd workers
node services/smart-service-detector.js
cd ..

echo.
echo 📦 Step 2/5: Setup Missing Services
echo ==================================
call setup-windows-native-smart.bat

echo.
echo 🚀 Step 3/5: Starting All Services
echo =================================
call start-all-services-smart.bat

echo Waiting for services to fully initialize...
timeout /t 15 /nobreak

echo.
echo 🗄️ Step 4/5: Database Configuration
echo ==================================
call setup-database-smart.bat

echo.
echo 🧪 Step 5/5: System Verification
echo ===============================
call test-system-smart.bat

echo.
echo 🎉 DEMO COMPLETE!
echo ================

echo.
echo 📋 Your Evidence Processing System is now running:
echo.
echo 🌐 Web Interfaces:
netstat -an | findstr ":15672" >nul 2>&1 && echo   ✅ RabbitMQ Management: http://localhost:15672 (guest/guest)
netstat -an | findstr ":7474" >nul 2>&1 && echo   ✅ Neo4j Browser: http://localhost:7474 (neo4j/neo4j - change password on first login)
netstat -an | findstr ":9001" >nul 2>&1 && echo   ✅ MinIO Console: http://localhost:9001 (evidence/evidence123)
netstat -an | findstr ":6333" >nul 2>&1 && echo   ✅ Qdrant Dashboard: http://localhost:6333/dashboard

echo.
echo 🏭 To Start Evidence Processing:
echo   run: start-worker.bat
echo.
echo 🛑 To Stop Everything:
echo   run: stop-all-services.bat
echo.
echo 📊 To Test Again:
echo   run: test-system-smart.bat
echo.
echo 💡 The system automatically detected and used:
echo   • Your existing PostgreSQL (password: 123456)
echo   • System Redis/RabbitMQ (if available)
echo   • Downloaded portable services as needed
echo.

echo Press any key to start the worker and begin processing evidence...
pause

echo.
echo 🏭 Starting Evidence Processing Worker...
echo ========================================
call start-worker.bat
