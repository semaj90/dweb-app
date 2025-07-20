@echo off
echo 🚀 Starting Legal AI Assistant with PostgreSQL...

echo.
echo 1️⃣ Setting environment to PostgreSQL...
copy /Y .env.testing .env

echo.
echo 2️⃣ Starting Docker services...
docker-compose up -d postgres redis qdrant

echo.
echo 3️⃣ Waiting for services to start (10 seconds)...
timeout /t 10 /nobreak > nul

echo.
echo 4️⃣ Running database migrations...
npx drizzle-kit push

echo.
echo 5️⃣ Initializing database...
node init-postgres.js

echo.
echo 6️⃣ Starting development server...
npm run dev

pause
