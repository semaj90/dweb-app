@echo off
set PGPASSWORD=123456
echo 📊 Checking legal_documents table schema...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "\d legal_documents" -q