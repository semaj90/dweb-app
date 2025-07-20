@echo off
echo Starting Legal AI Assistant Platform...

echo.
echo Starting PostgreSQL and Redis...
docker compose up -d postgres redis

echo.
echo Waiting for database...
timeout /t 5 /nobreak > nul

echo.
echo Running database migrations...
call npm run db:migrate

echo.
echo Starting frontend server...
call npm run dev
