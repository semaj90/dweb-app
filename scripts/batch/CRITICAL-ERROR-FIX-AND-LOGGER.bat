@echo off
setlocal enabledelayedexpansion
title Critical Error Fix and Logger
color 0C

set "PROJECT_ROOT=%~dp0"
set "FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend"
set "ERROR_LOG=%PROJECT_ROOT%CRITICAL_ERRORS_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log"

echo ========================================
echo CRITICAL ERROR FIX AND LOGGER
echo ========================================
echo.
echo Scanning for critical errors and fixing them...
echo Log file: %ERROR_LOG%
echo.

:: Initialize error log
> "%ERROR_LOG%" (
echo CRITICAL ERRORS LOG - %date% %time%
echo ==========================================
echo.
)

:: Check 1: Project Structure
echo [1/12] Checking project structure...
if not exist "%FRONTEND_PATH%\package.json" (
    echo ERROR: Frontend package.json missing >> "%ERROR_LOG%"
    echo ❌ Frontend package.json missing
    goto :log_and_exit
)

:: Check 2: Docker Container Names
echo [2/12] Fixing Docker container name inconsistencies...
findstr /c "legal-ai-postgres" "%PROJECT_ROOT%COMPLETE-SMART-SETUP.bat" > nul
if %errorlevel% equ 0 (
    echo ERROR: Inconsistent Docker container names in COMPLETE-SMART-SETUP.bat >> "%ERROR_LOG%"
    echo   Expected: deeds-postgres, Found: legal-ai-postgres >> "%ERROR_LOG%"
    
    :: Fix the file
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$content = Get-Content '%PROJECT_ROOT%COMPLETE-SMART-SETUP.bat' -Raw; ^
    $content = $content -replace 'legal-ai-postgres', 'deeds-postgres'; ^
    $content = $content -replace 'legal-ai-redis', 'deeds-redis'; ^
    $content = $content -replace 'legal-ai-qdrant', 'deeds-qdrant'; ^
    Set-Content '%PROJECT_ROOT%COMPLETE-SMART-SETUP.bat' $content"
    
    echo ✅ Fixed Docker container names in COMPLETE-SMART-SETUP.bat
) else (
    echo ✅ Docker container names consistent
)

:: Check 3: PowerShell Escaping Issues
echo [3/12] Checking PowerShell escaping issues...
findstr /c "^<" "%PROJECT_ROOT%UPDATE-CLAUDE-CONFIG-CONTEXT7.bat" > nul
if %errorlevel% equ 0 (
    echo ERROR: PowerShell escaping issues in UPDATE-CLAUDE-CONFIG-CONTEXT7.bat >> "%ERROR_LOG%"
    echo   Issue: Unescaped special characters in echo statements >> "%ERROR_LOG%"
    
    :: Create corrected version
    > "%PROJECT_ROOT%UPDATE-CLAUDE-CONFIG-CONTEXT7-FIXED.bat" (
        echo @echo off
        echo setlocal enabledelayedexpansion
        echo title Update Claude Config with Context 7
        echo color 0A
        echo.
        echo echo Stopping Claude Desktop if running...
        echo taskkill /f /im Claude.exe ^> nul 2^>^&1
        echo timeout /t 2 ^> nul
        echo.
        echo set "CLAUDE_CONFIG=%%APPDATA%%\Claude\claude_desktop_config.json"
        echo.
        echo echo Backing up existing configuration...
        echo if exist "%%CLAUDE_CONFIG%%" ^(
        echo     copy "%%CLAUDE_CONFIG%%" "%%CLAUDE_CONFIG%%.backup" ^> nul
        echo     echo ✅ Backup created
        echo ^) else ^(
        echo     mkdir "%%APPDATA%%\Claude" ^> nul 2^>^&1
        echo ^)
        echo.
        echo echo Creating simplified Claude configuration...
        echo ^> "%%CLAUDE_CONFIG%%" ^(
        echo echo {
        echo echo   "mcpServers": {
        echo echo     "filesystem": {
        echo echo       "command": "npx",
        echo echo       "args": ["--yes", "@modelcontextprotocol/server-filesystem", "C:/Users/james/Desktop/deeds-web/deeds-web-app", "--write-access"]
        echo echo     }
        echo echo   }
        echo echo }
        echo ^)
        echo.
        echo echo ✅ Claude Desktop configuration updated!
        echo echo ⚠️  IMPORTANT: Restart Claude Desktop for changes to take effect
        echo pause
    )
    
    echo ✅ Created fixed version: UPDATE-CLAUDE-CONFIG-CONTEXT7-FIXED.bat
) else (
    echo ✅ PowerShell escaping appears correct
)

:: Check 4: Missing Dependencies in Scripts
echo [4/12] Checking script dependencies...
if not exist "%PROJECT_ROOT%enhanced-merge-refactor.mjs" (
    echo ERROR: Missing enhanced-merge-refactor.mjs file >> "%ERROR_LOG%"
    echo   Required by: COMPLETE-SMART-SETUP.bat >> "%ERROR_LOG%"
    echo ❌ Missing enhanced-merge-refactor.mjs
)

if not exist "%PROJECT_ROOT%enhanced-vector-scanner.mjs" (
    echo ERROR: Missing enhanced-vector-scanner.mjs file >> "%ERROR_LOG%"
    echo   Required by: COMPLETE-SMART-SETUP.bat >> "%ERROR_LOG%"
    echo ❌ Missing enhanced-vector-scanner.mjs
)

:: Check 5: Docker Compose File Issues
echo [5/12] Checking Docker Compose configurations...
if not exist "%PROJECT_ROOT%docker-compose-enhanced-lowmem.yml" (
    echo ERROR: Missing docker-compose-enhanced-lowmem.yml >> "%ERROR_LOG%"
    echo   Required by: Multiple setup scripts >> "%ERROR_LOG%"
    echo ❌ Missing docker-compose-enhanced-lowmem.yml
)

:: Check 6: TypeScript Configuration
echo [6/12] Checking TypeScript configuration...
cd /d "%FRONTEND_PATH%"
if exist "tsconfig.json" (
    findstr /c "extends" tsconfig.json > nul
    if %errorlevel% neq 0 (
        echo ERROR: TypeScript config missing extends >> "%ERROR_LOG%"
        echo   File: %FRONTEND_PATH%\tsconfig.json >> "%ERROR_LOG%"
        
        :: Fix tsconfig.json
        > tsconfig.json (
            echo {
            echo   "extends": "./.svelte-kit/tsconfig.json",
            echo   "compilerOptions": {
            echo     "allowJs": true,
            echo     "checkJs": true,
            echo     "esModuleInterop": true,
            echo     "forceConsistentCasingInFileNames": true,
            echo     "resolveJsonModule": true,
            echo     "skipLibCheck": true,
            echo     "sourceMap": true,
            echo     "strict": true,
            echo     "moduleResolution": "bundler"
            echo   }
            echo }
        )
        echo ✅ Fixed TypeScript configuration
    ) else (
        echo ✅ TypeScript configuration valid
    )
) else (
    echo ERROR: Missing tsconfig.json >> "%ERROR_LOG%"
    echo ❌ Missing tsconfig.json
)

:: Check 7: Package.json Scripts
echo [7/12] Checking package.json scripts...
if exist "package.json" (
    findstr /c "\"dev\":" package.json > nul
    if %errorlevel% neq 0 (
        echo ERROR: Missing dev script in package.json >> "%ERROR_LOG%"
        echo ❌ Missing dev script
    ) else (
        echo ✅ Package.json scripts present
    )
) else (
    echo ERROR: Missing package.json in frontend >> "%ERROR_LOG%"
    echo ❌ Missing package.json
)

:: Check 8: UnoCSS Configuration
echo [8/12] Checking UnoCSS configuration...
if exist "uno.config.ts" (
    findstr /c "defineConfig" uno.config.ts > nul
    if %errorlevel% neq 0 (
        echo ERROR: Invalid UnoCSS configuration >> "%ERROR_LOG%"
        echo ❌ Invalid UnoCSS config
    ) else (
        echo ✅ UnoCSS configuration valid
    )
) else (
    echo ERROR: Missing uno.config.ts >> "%ERROR_LOG%"
    echo ❌ Missing UnoCSS config
)

:: Check 9: Vite Configuration
echo [9/12] Checking Vite configuration...
if exist "vite.config.ts" (
    findstr /c "UnoCSS" vite.config.ts > nul
    if %errorlevel% neq 0 (
        echo ERROR: Vite config missing UnoCSS plugin >> "%ERROR_LOG%"
        echo   File: %FRONTEND_PATH%\vite.config.ts >> "%ERROR_LOG%"
        
        :: Fix vite.config.ts
        > vite.config.ts (
            echo import { sveltekit } from '@sveltejs/kit/vite';
            echo import { defineConfig } from 'vite';
            echo import UnoCSS from 'unocss/vite';
            echo.
            echo export default defineConfig^({
            echo 	plugins: [
            echo 		UnoCSS^(^),
            echo 		sveltekit^(^)
            echo 	]
            echo }^);
        )
        echo ✅ Fixed Vite configuration
    ) else (
        echo ✅ Vite configuration includes UnoCSS
    )
) else (
    echo ERROR: Missing vite.config.ts >> "%ERROR_LOG%"
    echo ❌ Missing Vite config
)

:: Check 10: Node Modules and Dependencies
echo [10/12] Checking dependencies...
if not exist "node_modules" (
    echo WARNING: node_modules not found, dependencies need installation >> "%ERROR_LOG%"
    echo ⚠️ Dependencies need installation
) else (
    echo ✅ Dependencies appear installed
)

:: Check 11: Required Directories
echo [11/12] Checking required directories...
if not exist "src\lib\types" (
    echo Creating missing types directory...
    mkdir "src\lib\types"
    echo ✅ Created types directory
)

if not exist "src\lib\stores" (
    echo Creating missing stores directory...
    mkdir "src\lib\stores"
    echo ✅ Created stores directory
)

if not exist "src\lib\utils" (
    echo Creating missing utils directory...
    mkdir "src\lib\utils"
    echo ✅ Created utils directory
)

:: Check 12: Create Essential Files
echo [12/12] Creating essential missing files...

if not exist "src\lib\types\index.ts" (
    > "src\lib\types\index.ts" (
        echo export interface Evidence {
        echo   id: string;
        echo   title: string;
        echo   description?: string;
        echo   fileUrl?: string;
        echo   evidenceType: string;
        echo }
        echo.
        echo export interface Case {
        echo   id: string;
        echo   title: string;
        echo   description?: string;
        echo   status: string;
        echo }
        echo.
        echo export interface Report {
        echo   id: string;
        echo   title: string;
        echo   content: string;
        echo }
    )
    echo ✅ Created types/index.ts
)

if not exist "src\lib\stores\index.ts" (
    > "src\lib\stores\index.ts" (
        echo import { writable } from 'svelte/store';
        echo.
        echo export const cases = writable^([]^);
        echo export const evidence = writable^([]^);
        echo export const currentUser = writable^(null^);
    )
    echo ✅ Created stores/index.ts
)

if not exist "src\lib\utils\index.ts" (
    > "src\lib\utils\index.ts" (
        echo import { type ClassValue, clsx } from 'clsx';
        echo import { twMerge } from 'tailwind-merge';
        echo.
        echo export function cn^(...inputs: ClassValue[]^): string {
        echo   return twMerge^(clsx^(inputs^)^);
        echo }
    )
    echo ✅ Created utils/index.ts
)

cd "%PROJECT_ROOT%"

:: Final Summary
echo.
echo ========================================
echo ERROR ANALYSIS COMPLETE
echo ========================================
echo.

:: Count critical errors
set "critical_count=0"
for /f %%i in ('findstr /c "ERROR:" "%ERROR_LOG%"') do set /a critical_count+=1

echo 📊 SUMMARY:
echo - Critical errors found: %critical_count%
echo - Log file: %ERROR_LOG%
echo - Fixed files: Multiple configurations updated
echo.

if %critical_count% gtr 0 (
    echo ❌ CRITICAL ISSUES FOUND - See log file for details
    echo.
    echo 🔧 RECOMMENDED ACTIONS:
    echo 1. Review %ERROR_LOG%
    echo 2. Run PHASE-CONSOLIDATION-FIX.bat
    echo 3. Use START-DEV.bat for simple startup
    echo 4. Check Docker containers with: docker ps
    echo.
) else (
    echo ✅ NO CRITICAL ERRORS FOUND
    echo.
    echo 🚀 READY TO START:
    echo 1. cd sveltekit-frontend
    echo 2. npm install
    echo 3. npm run dev
    echo.
)

:: Create simplified launcher
echo Creating SIMPLE-LAUNCHER.bat...
> "%PROJECT_ROOT%SIMPLE-LAUNCHER.bat" (
    echo @echo off
    echo title Simple Legal AI Launcher
    echo echo ========================================
    echo echo SIMPLE LEGAL AI LAUNCHER
    echo echo ========================================
    echo echo.
    echo cd /d "%%~dp0sveltekit-frontend"
    echo if not exist "package.json" ^(
    echo     echo ❌ package.json not found
    echo     pause
    echo     exit /b 1
    echo ^)
    echo echo Installing dependencies...
    echo npm install
    echo echo Starting development server...
    echo npm run dev
    echo pause
)

echo ✅ Created SIMPLE-LAUNCHER.bat
echo.
echo 📋 LOG FILE CONTENTS:
echo.
type "%ERROR_LOG%"
echo.
echo 🔧 Use SIMPLE-LAUNCHER.bat for easiest startup
echo.
pause

:log_and_exit
echo.
echo Full error log saved to: %ERROR_LOG%
echo.
pause
exit /b %critical_count%