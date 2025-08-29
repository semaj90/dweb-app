-- Create missing keys table for legal AI database
-- This table handles authentication keys, API keys, and password management

CREATE TABLE IF NOT EXISTS "keys" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "user_id" uuid NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "hashed_password" varchar(255),
  "key_type" varchar(50) NOT NULL,
  "key_id" varchar(255) NOT NULL,
  "key_value" text,
  "expires_at" timestamp,
  "is_active" boolean DEFAULT true NOT NULL,
  "metadata" jsonb DEFAULT '{}' NOT NULL,
  "created_at" timestamp DEFAULT now() NOT NULL,
  "updated_at" timestamp DEFAULT now() NOT NULL
);

-- Create indexes for the keys table
CREATE INDEX IF NOT EXISTS "keys_user_id_idx" ON "keys" USING btree ("user_id");
CREATE UNIQUE INDEX IF NOT EXISTS "keys_key_id_idx" ON "keys" USING btree ("key_id");
CREATE INDEX IF NOT EXISTS "keys_key_type_idx" ON "keys" USING btree ("key_type");
CREATE INDEX IF NOT EXISTS "keys_active_idx" ON "keys" USING btree ("is_active");

-- Create updated_at trigger for keys table
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_keys_updated_at ON "keys";
CREATE TRIGGER update_keys_updated_at
    BEFORE UPDATE ON "keys"
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add default keys for existing users (if any)
-- This will create password-type keys for users who don't have any keys yet
INSERT INTO "keys" ("user_id", "key_type", "key_id", "hashed_password", "is_active")
SELECT 
  u."id" as "user_id",
  'password' as "key_type",
  u."email" || ':password' as "key_id",
  u."hashed_password",
  u."is_active"
FROM "users" u
WHERE u."hashed_password" IS NOT NULL 
  AND NOT EXISTS (
    SELECT 1 FROM "keys" k WHERE k."user_id" = u."id" AND k."key_type" = 'password'
  );

-- Create function to automatically create password key when user is created
CREATE OR REPLACE FUNCTION create_user_password_key()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.hashed_password IS NOT NULL THEN
        INSERT INTO "keys" ("user_id", "key_type", "key_id", "hashed_password", "is_active")
        VALUES (
            NEW.id,
            'password',
            NEW.email || ':password',
            NEW.hashed_password,
            NEW.is_active
        );
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-create password key for new users
DROP TRIGGER IF EXISTS auto_create_user_password_key ON "users";
CREATE TRIGGER auto_create_user_password_key
    AFTER INSERT ON "users"
    FOR EACH ROW
    EXECUTE FUNCTION create_user_password_key();

-- Create function to sync password changes between users and keys
CREATE OR REPLACE FUNCTION sync_user_password_to_keys()
RETURNS TRIGGER AS $$
BEGIN
    -- Update existing password key
    UPDATE "keys" 
    SET 
        "hashed_password" = NEW.hashed_password,
        "is_active" = NEW.is_active,
        "updated_at" = now()
    WHERE "user_id" = NEW.id AND "key_type" = 'password';
    
    -- If no password key exists and we have a password, create one
    IF NEW.hashed_password IS NOT NULL AND NOT FOUND THEN
        INSERT INTO "keys" ("user_id", "key_type", "key_id", "hashed_password", "is_active")
        VALUES (
            NEW.id,
            'password',
            NEW.email || ':password',
            NEW.hashed_password,
            NEW.is_active
        );
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to sync password updates
DROP TRIGGER IF EXISTS sync_user_password_to_keys ON "users";
CREATE TRIGGER sync_user_password_to_keys
    AFTER UPDATE ON "users"
    FOR EACH ROW
    WHEN (OLD.hashed_password IS DISTINCT FROM NEW.hashed_password OR OLD.is_active IS DISTINCT FROM NEW.is_active)
    EXECUTE FUNCTION sync_user_password_to_keys();

-- Verify the table was created successfully
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'keys' AND table_schema = 'public') THEN
        RAISE NOTICE 'SUCCESS: Keys table created successfully';
        RAISE NOTICE 'Keys table has % rows', (SELECT COUNT(*) FROM keys);
    ELSE
        RAISE EXCEPTION 'FAILED: Keys table was not created';
    END IF;
END
$$;