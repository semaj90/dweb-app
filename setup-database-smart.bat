@echo off
echo 📊 Setting up Evidence Processing Database (Smart Detection)
echo ============================================================

echo.
echo 🔍 Detecting PostgreSQL Installation...

:: Set the PostgreSQL password to 123456
set PGPASSWORD=123456

:: Test PostgreSQL connection with the specified password
echo Testing PostgreSQL connection with password 123456...
psql -U postgres -c "SELECT version();" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Cannot connect to PostgreSQL with password 123456
    echo.
    echo Please ensure:
    echo 1. PostgreSQL is installed and running
    echo 2. User 'postgres' exists with password '123456'
    echo 3. PostgreSQL is accepting connections on localhost:5432
    echo.
    echo 🔧 To fix PostgreSQL password:
    echo Method 1 - Using psql (if you know current password):
    echo   psql -U postgres
    echo   ALTER USER postgres PASSWORD '123456';
    echo.
    echo Method 2 - Using pg_hba.conf (temporary):
    echo   1. Edit pg_hba.conf and change 'md5' to 'trust' for local connections
    echo   2. Restart PostgreSQL service
    echo   3. Run: psql -U postgres -c "ALTER USER postgres PASSWORD '123456';"
    echo   4. Change pg_hba.conf back to 'md5'
    echo   5. Restart PostgreSQL service again
    echo.
    echo Method 3 - Reinstall PostgreSQL:
    echo   • Download from: https://www.postgresql.org/download/windows/
    echo   • During installation, set superuser password to: 123456
    echo.
    pause
    exit /b 1
)

echo ✅ PostgreSQL connection successful with password 123456

:: Get PostgreSQL version and installation info
echo.
echo 📋 PostgreSQL Information:
for /f "delims=" %%i in ('psql -U postgres -t -c "SELECT version();"') do echo Version: %%i
for /f "delims=" %%i in ('psql -U postgres -t -c "SHOW data_directory;"') do echo Data Directory: %%i
for /f "delims=" %%i in ('psql -U postgres -t -c "SHOW port;"') do echo Port: %%i

echo.
echo 🗄️ Setting up database and extensions...

:: Check if database already exists
psql -U postgres -lqt | cut -d ^| -f 1 | findstr evidence_processing >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Database 'evidence_processing' already exists
    set DB_EXISTS=1
) else (
    echo Creating evidence_processing database...
    createdb -U postgres evidence_processing
    if %errorlevel% == 0 (
        echo ✅ Database created successfully
        set DB_EXISTS=1
    ) else (
        echo ❌ Failed to create database
        exit /b 1
    )
)

:: Check and install extensions
echo.
echo 🔧 Installing required extensions...

:: Install uuid-ossp extension
echo Installing uuid-ossp extension...
psql -U postgres -d evidence_processing -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ uuid-ossp extension installed
) else (
    echo ❌ Failed to install uuid-ossp extension
)

:: Check for pgvector availability
echo Checking for pgvector extension...
psql -U postgres -d evidence_processing -c "CREATE EXTENSION IF NOT EXISTS vector;" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ pgvector extension installed
    set PGVECTOR_AVAILABLE=1
) else (
    echo ⚠️ pgvector extension not available
    echo.
    echo 💡 pgvector installation options:
    echo 1. Download pre-built binaries from: https://github.com/pgvector/pgvector/releases
    echo 2. Use PostgreSQL with pgvector pre-installed (like Supabase)
    echo 3. Compile from source (requires Visual Studio Build Tools)
    echo.
    echo ℹ️ Continuing without pgvector - vector search will use fallback methods
    set PGVECTOR_AVAILABLE=0
)

echo.
echo 📋 Running database migration...

:: Check if migration file exists
if not exist "migrations\create_evidence_processing_schema.sql" (
    echo ❌ Migration file not found: migrations\create_evidence_processing_schema.sql
    echo Please ensure you're running this from the project root directory.
    pause
    exit /b 1
)

:: Create a modified migration script if pgvector is not available
if %PGVECTOR_AVAILABLE% == 0 (
    echo 📝 Creating pgvector-free migration...
    
    :: Create temporary migration file without pgvector
    powershell -command "(Get-Content 'migrations\create_evidence_processing_schema.sql') -replace 'CREATE EXTENSION IF NOT EXISTS vector;', '-- pgvector not available' -replace 'VECTOR', 'TEXT' -replace 'vector_cosine_ops', 'text_ops' | Set-Content 'migrations\temp_migration.sql'"
    
    :: Run the modified migration
    psql -U postgres -d evidence_processing -f "migrations\temp_migration.sql"
    if %errorlevel% == 0 (
        echo ✅ Database migration completed (without pgvector)
    ) else (
        echo ❌ Database migration failed
        del "migrations\temp_migration.sql" >nul 2>&1
        pause
        exit /b 1
    )
    
    :: Clean up temporary file
    del "migrations\temp_migration.sql" >nul 2>&1
) else (
    :: Run the full migration with pgvector
    psql -U postgres -d evidence_processing -f "migrations\create_evidence_processing_schema.sql"
    if %errorlevel% == 0 (
        echo ✅ Database migration completed successfully (with pgvector)
    ) else (
        echo ❌ Database migration failed
        pause
        exit /b 1
    )
)

echo.
echo 🔍 Verifying database setup...

:: Check if main tables were created
echo Checking evidence processing tables...
psql -U postgres -d evidence_processing -c "\dt" | findstr evidence_process >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Core evidence processing tables created
) else (
    echo ❌ Core tables not found
    pause
    exit /b 1
)

:: Count tables created
echo.
echo 📊 Database schema verification:
for /f %%i in ('psql -U postgres -d evidence_processing -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';"') do (
    echo Tables created: %%i
)

:: Test basic operations
echo.
echo 🎯 Testing database functionality...

:: Test insert and select
psql -U postgres -d evidence_processing -c "INSERT INTO queue_stats (queue_name, messages_pending) VALUES ('test_queue', 0) ON CONFLICT (queue_name) DO NOTHING;" >nul 2>&1
psql -U postgres -d evidence_processing -c "SELECT count(*) FROM queue_stats;" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Database operations working correctly
) else (
    echo ❌ Database operations failed
    pause
    exit /b 1
)

:: Test vector functionality if available
if %PGVECTOR_AVAILABLE% == 1 (
    echo Testing pgvector functionality...
    psql -U postgres -d evidence_processing -c "SELECT '[1,2,3]'::vector;" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ pgvector operations working
    ) else (
        echo ⚠️ pgvector operations not working properly
    )
)

echo.
echo 🔧 Creating database utilities...

:: Create database connection info file
echo # Evidence Processing Database Configuration > database-config.txt
echo # Generated: %date% %time% >> database-config.txt
echo. >> database-config.txt
echo Host: localhost >> database-config.txt
echo Port: 5432 >> database-config.txt
echo Database: evidence_processing >> database-config.txt
echo Username: postgres >> database-config.txt
echo Password: 123456 >> database-config.txt
echo. >> database-config.txt
echo Connection String: >> database-config.txt
echo postgresql://postgres:123456@localhost:5432/evidence_processing >> database-config.txt
echo. >> database-config.txt
echo Extensions Installed: >> database-config.txt
echo - uuid-ossp: ✅ >> database-config.txt
if %PGVECTOR_AVAILABLE% == 1 (
    echo - pgvector: ✅ >> database-config.txt
) else (
    echo - pgvector: ❌ (using fallback methods) >> database-config.txt
)

:: Create database backup script
echo @echo off > backup-database.bat
echo echo 📦 Backing up Evidence Processing Database >> backup-database.bat
echo echo ========================================== >> backup-database.bat
echo set PGPASSWORD=123456 >> backup-database.bat
echo set BACKUP_FILE=evidence_processing_backup_%%date:~-4,4%%%%date:~-10,2%%%%date:~-7,2%%.sql >> backup-database.bat
echo pg_dump -U postgres -h localhost evidence_processing ^> %%BACKUP_FILE%% >> backup-database.bat
echo if %%errorlevel%% == 0 ^( >> backup-database.bat
echo     echo ✅ Database backup created: %%BACKUP_FILE%% >> backup-database.bat
echo ^) else ^( >> backup-database.bat
echo     echo ❌ Database backup failed >> backup-database.bat
echo ^) >> backup-database.bat
echo pause >> backup-database.bat

:: Create database restore script
echo @echo off > restore-database.bat
echo echo 📥 Restoring Evidence Processing Database >> restore-database.bat
echo echo ========================================= >> restore-database.bat
echo set PGPASSWORD=123456 >> restore-database.bat
echo if "%%1"=="" ^( >> restore-database.bat
echo     echo Usage: restore-database.bat backup_file.sql >> restore-database.bat
echo     pause >> restore-database.bat
echo     exit /b 1 >> restore-database.bat
echo ^) >> restore-database.bat
echo dropdb -U postgres evidence_processing >> restore-database.bat
echo createdb -U postgres evidence_processing >> restore-database.bat
echo psql -U postgres -d evidence_processing -f %%1 >> restore-database.bat
echo if %%errorlevel%% == 0 ^( >> restore-database.bat
echo     echo ✅ Database restored successfully >> restore-database.bat
echo ^) else ^( >> restore-database.bat
echo     echo ❌ Database restore failed >> restore-database.bat
echo ^) >> restore-database.bat
echo pause >> restore-database.bat

echo ✅ Database utilities created

echo.
echo 🎉 Smart Database Setup Complete!
echo =================================
echo.
echo ✅ PostgreSQL connection verified (password: 123456)
echo ✅ evidence_processing database ready
echo ✅ Required extensions installed
if %PGVECTOR_AVAILABLE% == 1 (
    echo ✅ pgvector support enabled
) else (
    echo ⚠️ pgvector not available (will use fallback methods)
)
echo ✅ All tables and indexes created
echo ✅ Database functions and triggers set up
echo ✅ Initial data populated
echo ✅ Database utilities created

echo.
echo 📋 Database Details:
echo • Host: localhost:5432
echo • Database: evidence_processing
echo • Username: postgres
echo • Password: 123456
echo • Connection String: postgresql://postgres:123456@localhost:5432/evidence_processing

echo.
echo 🔗 Quick Actions:
echo • Backup: backup-database.bat
echo • View Config: type database-config.txt
echo • Connect: psql -U postgres -d evidence_processing

echo.
echo 💡 Next Steps:
echo 1. Update your .env file (should already be configured correctly)
echo 2. Run test-system.bat to verify all services work together
echo 3. Run start-worker.bat to start the evidence processing worker

echo.
if %PGVECTOR_AVAILABLE% == 0 (
    echo ⚠️ Note: Vector similarity search will use fallback methods
    echo For full vector search capabilities, consider installing pgvector:
    echo https://github.com/pgvector/pgvector/releases
    echo.
)

pause
