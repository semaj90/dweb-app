@echo off
echo SvelteKit Error Check and Fix
echo ==============================
echo.

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo 📊 Current error status BEFORE fixes:
echo ====================================
npm run check > before-fix-errors.txt 2>&1
type before-fix-errors.txt | findstr /i "error\|warning\|found"

echo.
echo 🔧 Applying SvelteKit best practices fixes...
echo ============================================
node sveltekit-best-practices-fix.mjs

echo.
echo 📊 Error status AFTER fixes:
echo ===========================
npm run check > after-fix-errors.txt 2>&1
type after-fix-errors.txt | findstr /i "error\|warning\|found"

echo.
echo 📋 Comparing before and after:
echo ==============================
echo BEFORE:
type before-fix-errors.txt | findstr /i "found"
echo.
echo AFTER: 
type after-fix-errors.txt | findstr /i "found"

echo.
echo ✅ Check completed! Review the output above.
echo 📄 Full logs saved to before-fix-errors.txt and after-fix-errors.txt
echo.

if exist "SVELTEKIT_FIXES_REPORT.md" (
    echo 📋 Fix report generated: SVELTEKIT_FIXES_REPORT.md
    echo.
)

echo 🚀 Next steps:
echo - If errors under 300: try npm run dev
echo - Use Find in Files for manual patterns
echo - Focus on highest error count files
echo.
pause
