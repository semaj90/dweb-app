@echo off
REM Test the Simple Vector Service
echo ====================================
echo Testing Simple Vector Service
echo ====================================

echo Starting service...
start /MIN "VectorService" bin\simple-vector-service.exe

echo Waiting for service to start...
timeout /t 3 > nul

echo Testing health endpoint...
curl -s http://localhost:8095/api/health

echo.
echo Testing vector operation...
curl -s -X POST http://localhost:8095/api/vector ^
  -H "Content-Type: application/json" ^
  -d "{\"request_id\":\"test-1\",\"vector\":[1,2,3,4],\"operation\":\"normalize\"}"

echo.
echo ====================================
echo Test completed. Check results above.
echo Service is running at http://localhost:8095
echo ====================================
pause