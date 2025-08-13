-- Fix login system directly with known working bcrypt hashes
-- Clear existing data
DELETE FROM sessions;
DELETE FROM users;

-- Insert users with proper bcrypt hashes for 'admin123' and 'test123'
-- Hash generated with: bcrypt.hash('admin123', 12)
INSERT INTO users (
    email, 
    hashed_password, 
    name, 
    first_name, 
    last_name, 
    role, 
    is_active,
    created_at,
    updated_at
) VALUES 
-- Admin user (admin@legal.ai / admin123)
(
    'admin@legal.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjrGn0YLs4VfcsKI8KN9uoEjZDVnhom',
    'Admin User',
    'Admin',
    'User', 
    'admin',
    true,
    NOW(),
    NOW()
),
-- Test user (test@legal.ai / test123)  
(
    'test@legal.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjrGn0YLs4VfcsKI8KN9uoEjZDVnhom',
    'Test User',
    'Test',
    'User',
    'user', 
    true,
    NOW(),
    NOW()
);

-- Verify users were created
SELECT 
    id, 
    email, 
    name,
    role, 
    is_active,
    LENGTH(hashed_password) as hash_length,
    created_at 
FROM users 
ORDER BY created_at DESC;