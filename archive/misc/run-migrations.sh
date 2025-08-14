#!/bin/bash
# Run migrations for Legal AI system

echo "🗄️ Running Database Migrations..."

# Set environment variables
export PGPASSWORD="LegalAI2024!"
export DATABASE_URL="postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
sleep 10

# Run migration
echo "Executing migration..."
psql -h localhost -p 5432 -U legal_admin -d legal_ai_db -f database/migrations/001_initial_schema.sql

if [ $? -eq 0 ]; then
    echo "✅ Migration completed successfully"
else
    echo "❌ Migration failed"
    exit 1
fi

# Test database connection
echo "Testing database connection..."
psql -h localhost -p 5432 -U legal_admin -d legal_ai_db -c "SELECT 'Database ready' as status;"

echo "Migration complete!"