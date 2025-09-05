@echo off
title Fix Svelte HTML Attributes
echo =======================================
echo Svelte HTML Attribute Fixer
echo =======================================
echo.
echo This script will fix common issues:
echo - className to class
echo - React event handlers to Svelte
echo - Template placeholders ${1} to actual CSS
echo - Broken ui-demo page
echo.

if not exist "src" (
    echo [ERROR] src directory not found. Run this from sveltekit-frontend directory.
    pause
    exit /b 1
)

echo [INFO] Starting HTML attribute fixes...
echo.

node fix-html-attributes.mjs

echo.
echo [SUCCESS] HTML attribute fixing completed!
echo.
echo What was fixed:
echo - All className attributes changed to class
echo - React event handlers converted to Svelte syntax
echo - Template placeholders replaced with proper CSS classes
echo - ui-demo page completely rebuilt with proper Svelte syntax
echo.
echo Backup files (.backup) have been created for changed files.
echo.
pause
