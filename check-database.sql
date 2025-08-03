-- Check PostgreSQL and database
SELECT version();
SELECT current_database();
SELECT current_user;

-- List all tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check users table structure
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'users'
ORDER BY ordinal_position;

-- Count users
SELECT COUNT(*) as user_count FROM users;

-- Show sample users (without passwords)
SELECT id, email, name, role, is_active, created_at 
FROM users 
LIMIT 5;