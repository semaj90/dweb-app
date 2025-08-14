-- Seed demo users for legal AI system
-- Note: Using bcrypt hashes for password 'admin123', 'prosecutor123', 'detective123', 'user123'

-- Admin user (admin@legal.ai / admin123)
INSERT INTO users (id, email, password_hash, first_name, last_name, role, is_active, created_at) 
VALUES (
    uuid_generate_v4(),
    'admin@legal.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjG7rGx4MV9g.dPt3/e3c4S6BqZjdJK', -- admin123
    'Admin',
    'User',
    'admin',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Prosecutor user (prosecutor@legal.ai / prosecutor123)
INSERT INTO users (id, email, password_hash, first_name, last_name, role, is_active, created_at) 
VALUES (
    uuid_generate_v4(),
    'prosecutor@legal.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjG7rGx4MV9g.dPt3/e3c4S6BqZjdJK', -- prosecutor123
    'Sarah',
    'Johnson',
    'prosecutor',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Detective user (detective@legal.ai / detective123)
INSERT INTO users (id, email, password_hash, first_name, last_name, role, is_active, created_at) 
VALUES (
    uuid_generate_v4(),
    'detective@legal.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjG7rGx4MV9g.dPt3/e3c4S6BqZjdJK', -- detective123
    'Mike',
    'Rodriguez',
    'detective',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Analyst user (analyst@legal.ai / user123)
INSERT INTO users (id, email, password_hash, first_name, last_name, role, is_active, created_at) 
VALUES (
    uuid_generate_v4(),
    'analyst@legal.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjG7rGx4MV9g.dPt3/e3c4S6BqZjdJK', -- user123
    'Emma',
    'Chen',
    'user',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Verify users were created
SELECT id, email, first_name, last_name, role, is_active, created_at 
FROM users 
ORDER BY created_at DESC;