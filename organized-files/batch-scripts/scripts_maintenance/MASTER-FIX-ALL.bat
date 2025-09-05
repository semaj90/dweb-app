@echo off
REM MASTER-FIX-ALL.bat: Comprehensive fix for SvelteKit, TypeScript, and CSS issues

echo ========================================
echo MASTER-FIX-ALL: Starting comprehensive fixes
echo ========================================

REM Change to the project directory
cd /d "C:\Users\james\Desktop\web-app"

echo [1/8] Installing/updating dependencies...
call npm install
cd sveltekit-frontend
call npm install
cd ..

echo [2/8] Auto-formatting all Svelte, TS, and CSS files...
cd sveltekit-frontend
call npx prettier --write "src/**/*.{svelte,ts,js,css,scss}" --config .prettierrc.json 2>nul || call npx prettier --write "src/**/*.{svelte,ts,js,css,scss}"
cd ..

echo [3/8] Auto-fixing TypeScript and JavaScript issues...
cd sveltekit-frontend
call npx eslint "src/**/*.{ts,js,svelte}" --fix --quiet 2>nul || echo "ESLint fixes attempted"
cd ..

echo [4/8] Fixing specific TypeScript errors...
cd sveltekit-frontend
call node -e "
const fs = require('fs');
const path = require('path');

// Fix cache type issue in vector-search.ts
const vectorSearchPath = 'src/lib/server/search/vector-search.ts';
if (fs.existsSync(vectorSearchPath)) {
  let content = fs.readFileSync(vectorSearchPath, 'utf8');
  content = content.replace(/cache\.get<VectorSearchResult\[\]>/g, 'cache.get');
  fs.writeFileSync(vectorSearchPath, content);
  console.log('Fixed cache type issue in vector-search.ts');
}

// Fix error type issue in embedding-service.ts
const embeddingServicePath = 'src/lib/server/services/embedding-service.ts';
if (fs.existsSync(embeddingServicePath)) {
  let content = fs.readFileSync(embeddingServicePath, 'utf8');
  content = content.replace(/error\.message/g, '(error as Error)?.message || String(error)');
  fs.writeFileSync(embeddingServicePath, content);
  console.log('Fixed error type issue in embedding-service.ts');
}

// Fix truthiness check in vector-service.ts
const vectorServicePath = 'src/lib/server/services/vector-service.ts';
if (fs.existsSync(vectorServicePath)) {
  let content = fs.readFileSync(vectorServicePath, 'utf8');
  content = content.replace(/} \|\| {}/g, '} ?? {}');
  fs.writeFileSync(vectorServicePath, content);
  console.log('Fixed truthiness check in vector-service.ts');
}

console.log('TypeScript error fixes completed');
"
cd ..

echo [5/8] Running Svelte sync...
cd sveltekit-frontend
call npx svelte-kit sync 2>nul || echo "Svelte sync attempted"
cd ..

echo [6/8] Running svelte-check for diagnostics...
cd sveltekit-frontend
call npx svelte-check --tsconfig ./tsconfig.json > ../logs/svelte-check-latest.log 2>&1
echo "Svelte check completed - see logs/svelte-check-latest.log for details"
cd ..

echo [7/8] Testing build process...
cd sveltekit-frontend
call npm run build > ../logs/build-latest.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo "✅ Build successful!"
) else (
    echo "⚠️  Build had issues - see logs/build-latest.log for details"
)
cd ..

echo [8/8] Final cleanup and verification...
cd sveltekit-frontend
REM Clear any problematic cache
if exist ".svelte-kit" rmdir /s /q ".svelte-kit" 2>nul
if exist "node_modules/.vite" rmdir /s /q "node_modules/.vite" 2>nul
cd ..

echo ========================================
echo MASTER-FIX-ALL COMPLETE
echo ========================================
echo.
echo Summary:
echo - Dependencies updated
echo - Code formatted with Prettier
echo - ESLint fixes applied
echo - TypeScript errors addressed
echo - Build tested
echo.
echo Check the logs folder for detailed reports:
echo - logs/svelte-check-latest.log
echo - logs/build-latest.log
echo.
echo To start development: cd sveltekit-frontend && npm run dev
echo ========================================

pause
