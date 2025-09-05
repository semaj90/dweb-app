@echo off

echo Starting Docker containers...
docker-compose up -d

echo Waiting for database to be ready...
:loop
docker-compose exec postgres pg_isready -U legal_admin -d legal_ai_phase34 >nul 2>&1
if %errorlevel% equ 0 goto db_ready
timeout /t 5 >nul
goto loop

:db_ready
echo Database is ready.

echo Running database migrations...
call npm run db:migrate

echo Starting SvelteKit development server...
call npm run dev
