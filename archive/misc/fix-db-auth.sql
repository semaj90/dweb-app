-- Reset legal_admin password and permissions
ALTER USER legal_admin PASSWORD '123456';
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO legal_admin;

-- Verify permissions
\du legal_admin
\l legal_ai_db