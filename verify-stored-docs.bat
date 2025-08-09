@echo off
set PGPASSWORD=123456
echo 📊 Checking stored documents...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT id, case_id, LEFT(content, 50) as content_preview, CASE WHEN embedding IS NOT NULL THEN 'HAS_EMBEDDING' ELSE 'NO_EMBEDDING' END as has_embedding FROM legal_documents LIMIT 5;" -q