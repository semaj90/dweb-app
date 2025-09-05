@echo off
echo 🚀 Legal AI Automated Codebase Improvement System
echo ==================================================
echo.

echo 📋 System Components:
echo   ✅ TypeScript Error Collection
echo   ✅ AI-Powered Error Analysis  
echo   ✅ Prioritized Fix Recommendations
echo   ✅ Progress Tracking & Metrics
echo   ✅ Automated Report Generation
echo.

echo 🔍 Step 1: Installing required dependencies...
npm install node-fetch --save-dev

echo.
echo 🤖 Step 2: Running AI-powered error analysis...
node automated-error-analysis.js

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Analysis completed successfully!
    echo.
    echo 📊 Generated outputs:
    echo   📁 ./error-reports/ - Detailed analysis reports
    echo   📈 ./error-history.json - Improvement tracking
    echo   📋 ./logs/ - Raw error logs
    echo.
    
    echo 🎯 Next steps:
    echo   1. Review the latest report in error-reports/
    echo   2. Address high-priority issues first  
    echo   3. Run this script regularly to track progress
    echo   4. Use npm run check to verify improvements
    
) else (
    echo ❌ Analysis failed - check the error logs
    echo.
    echo 🔧 Troubleshooting:
    echo   - Ensure Go service is running on port 8080
    echo   - Check network connectivity to /analyze endpoint
    echo   - Verify npm run check works correctly
)

echo.
echo 💡 Pro tip: Add this to your CI/CD pipeline for continuous improvement!
echo.
pause