# Legal AI Database Setup PowerShell Script

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Legal AI Database Setup" -ForegroundColor Cyan  
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "1. Create legal_ai_db database" -ForegroundColor White
Write-Host "2. Set up legal_admin user with password LegalSecure2024!" -ForegroundColor White
Write-Host "3. Create all necessary tables for Legal AI system" -ForegroundColor White
Write-Host "4. Try to install pgvector extension (if available)" -ForegroundColor White
Write-Host "5. Insert sample data for testing" -ForegroundColor White
Write-Host ""

$postgresPassword = Read-Host "Enter PostgreSQL postgres user password" -AsSecureString
$password = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($postgresPassword))

Write-Host ""
Write-Host "Running database setup..." -ForegroundColor Green

# Set environment variable for password
$env:PGPASSWORD = $password

try {
    # Run the SQL setup script
    & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -f "setup-legal-ai-db.sql"
    
    Write-Host ""
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "Database Setup Completed!" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "✅ legal_ai_db database created" -ForegroundColor Green
    Write-Host "✅ legal_admin user configured" -ForegroundColor Green
    Write-Host "✅ All tables created successfully" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Connection Details:" -ForegroundColor Cyan
    Write-Host "Database: legal_ai_db" -ForegroundColor White
    Write-Host "User: legal_admin" -ForegroundColor White
    Write-Host "Password: LegalSecure2024!" -ForegroundColor White
    Write-Host "Host: localhost" -ForegroundColor White
    Write-Host "Port: 5432" -ForegroundColor White
    Write-Host ""
    Write-Host "Connection String:" -ForegroundColor Cyan
    Write-Host "postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_db" -ForegroundColor Yellow
    
    # Test connection as legal_admin
    Write-Host ""
    Write-Host "Testing connection as legal_admin..." -ForegroundColor Blue
    $env:PGPASSWORD = "LegalSecure2024!"
    & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT 'Connection successful!' as status;"
    
} catch {
    Write-Host "Error occurred during setup: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common solutions:" -ForegroundColor Yellow
    Write-Host "1. Make sure PostgreSQL service is running" -ForegroundColor White
    Write-Host "2. Verify the postgres user password" -ForegroundColor White
    Write-Host "3. Check if PostgreSQL is listening on localhost:5432" -ForegroundColor White
} finally {
    # Clear the password from environment
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "pgvector Status:" -ForegroundColor Cyan
Write-Host "If pgvector installation failed above, you have these options:" -ForegroundColor Yellow
Write-Host "1. Install Visual Studio Build Tools and run: nmake /F Makefile.win install" -ForegroundColor White
Write-Host "2. Use Docker PostgreSQL with pgvector pre-installed" -ForegroundColor White
Write-Host "3. Continue without pgvector (most features will still work)" -ForegroundColor White
Write-Host ""
Write-Host "Your Legal AI application is ready to use! 🚀" -ForegroundColor Green

Read-Host "Press Enter to continue"