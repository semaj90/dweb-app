@echo off
REM Download and setup required services for Legal AI Platform
REM This batch script downloads the large binaries instead of storing them in git

echo 📦 Downloading Legal AI Platform Services...

REM Check if PowerShell is available and run the PowerShell script
where powershell >nul 2>&1
if %errorlevel% equ 0 (
    echo 🔧 Running PowerShell download script...
    powershell -ExecutionPolicy Bypass -File "scripts/download-services.ps1"
) else (
    echo ⚠️ PowerShell not found, using fallback curl/wget method...
    
    REM Fallback using curl if available
    where curl >nul 2>&1
    if %errorlevel% equ 0 (
        echo ⬇️ Downloading MinIO (~108MB)...
        if not exist "minio.exe" (
            curl -L -o minio.exe https://dl.min.io/server/minio/release/windows-amd64/minio.exe
            echo ✅ Downloaded MinIO
        ) else (
            echo ✅ minio.exe already exists, skipping...
        )
        
        echo ⬇️ Downloading Neo4j (~118MB)...
        if not exist "neo4j-community-5.23.0-windows.zip" (
            curl -L -o neo4j-community-5.23.0-windows.zip https://dist.neo4j.org/neo4j-community-5.23.0-windows.zip
            echo ✅ Downloaded Neo4j
        ) else (
            echo ✅ neo4j-community-5.23.0-windows.zip already exists, skipping...
        )
        
        echo ⬇️ Downloading Qdrant (~77MB)...
        if not exist "qdrant-windows.zip" (
            curl -L -o qdrant-windows.zip https://github.com/qdrant/qdrant/releases/download/v1.8.4/qdrant-x86_64-pc-windows-msvc.zip
            echo ✅ Downloaded Qdrant
        ) else (
            echo ✅ qdrant-windows.zip already exists, skipping...
        )
    ) else (
        echo ❌ Neither PowerShell nor curl found. Please install one of them.
        echo 💡 Alternatively, manually download:
        echo    - MinIO: https://dl.min.io/server/minio/release/windows-amd64/minio.exe
        echo    - Neo4j: https://dist.neo4j.org/neo4j-community-5.23.0-windows.zip  
        echo    - Qdrant: https://github.com/qdrant/qdrant/releases/download/v1.8.4/qdrant-x86_64-pc-windows-msvc.zip
        pause
        exit /b 1
    )
)

echo 🎉 Service download complete!
echo 💡 These files are automatically downloaded and not stored in git.
pause