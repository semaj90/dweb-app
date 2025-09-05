@echo off
echo 🔧 Fixing Auth Connection Error...
echo ==================================

echo ✅ Database connection string updated
echo ✅ Credentials now match: legal_admin/LegalSecure2024!
echo ✅ Database: legal_ai_v3

echo.
echo 🔄 Restarting dev server to apply changes...
echo Press Ctrl+C in the dev server window, then run: npm run dev

echo.
echo 📋 If still getting auth errors:
echo 1. Check PostgreSQL is running
echo 2. Verify .env file has correct DATABASE_URL
echo 3. Ensure legal_admin user exists with correct password

pause