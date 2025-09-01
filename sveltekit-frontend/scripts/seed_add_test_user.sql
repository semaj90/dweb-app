-- scripts/seed_add_test_user.sql
-- Purpose: Enable pgcrypto and insert a test user into the `users` table.
-- WARNING: Review the CREATE TABLE block below if you already have a users table to avoid schema conflicts.
-- DB connection example (PowerShell):
-- psql "postgresql://postgres:123456@localhost:5432/legal_ai_db" -f scripts/seed_add_test_user.sql

-- 1) Enable pgcrypto for server-side password hashing
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 2) Create a minimal users table if one does not already exist.
--    If you already have a production users table, remove or comment out this block.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'users'
  ) THEN
    CREATE TABLE public.users (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      email text NOT NULL UNIQUE,
      password_hash text NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    );
  END IF;
END$$;

-- 3) Insert test user with server-side bcrypt-style hashing (pgcrypto's crypt/gen_salt)
-- Replace email/password below as needed.
INSERT INTO public.users (email, password_hash)
VALUES (
  'test@example.com',
  crypt('secret123', gen_salt('bf'))
)
ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash;

-- 4) Verification queries (run manually or left for interactive check)
-- Select the inserted row:
-- SELECT id, email, password_hash, created_at FROM public.users WHERE email = 'test@example.com';

-- Check a plaintext password against stored hash (returns true/false):
-- SELECT (password_hash = crypt('secret123', password_hash)) AS password_ok
-- FROM public.users WHERE email = 'test@example.com';

-- End of script
