@echo off
set PGPASSWORD=123456
echo 🔧 Adding missing columns to legal_documents table...

"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS summary TEXT;" -q

"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS entities TEXT;" -q

"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS risk_score FLOAT DEFAULT 0;" -q

"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;" -q

echo ✅ Schema update completed

echo 📊 Updated table schema:
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "\d legal_documents" -q