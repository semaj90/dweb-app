-- Enhanced Authentication Schema Migration
-- Adds security features to users table and creates audit logging

-- Add new columns to users table for enhanced authentication
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS email_verified TIMESTAMP,
ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS password_reset_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS password_reset_expires TIMESTAMP,
ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS login_attempts INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS lockout_until TIMESTAMP,
ADD COLUMN IF NOT EXISTS two_factor_secret TEXT,
ADD COLUMN IF NOT EXISTS two_factor_enabled BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS profile_picture TEXT,
ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}';

-- Create sessions table for Lucia v3 compatibility
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  ip_address VARCHAR(45),
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create user audit logs table for security tracking
CREATE TABLE IF NOT EXISTS user_audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  action VARCHAR(100) NOT NULL,
  ip_address VARCHAR(45),
  user_agent TEXT,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_user_id ON user_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_action ON user_audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_created_at ON user_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_users_email_verification_token ON users(email_verification_token);
CREATE INDEX IF NOT EXISTS idx_users_password_reset_token ON users(password_reset_token);
CREATE INDEX IF NOT EXISTS idx_users_lockout_until ON users(lockout_until);

-- Add constraints
ALTER TABLE users 
ADD CONSTRAINT chk_login_attempts_positive CHECK (login_attempts >= 0),
ADD CONSTRAINT chk_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

-- Update existing users to have default preferences
UPDATE users 
SET preferences = '{
  "theme": "system",
  "notifications": true,
  "language": "en",
  "timezone": "UTC"
}'
WHERE preferences = '{}' OR preferences IS NULL;

-- Insert default security audit log for existing users
INSERT INTO user_audit_logs (user_id, action, ip_address, user_agent, metadata)
SELECT 
  id,
  'account_migrated',
  'system',
  'system',
  json_build_object(
    'migration_date', NOW(),
    'migration_version', '20250821_enhanced_auth_schema'
  )
FROM users
WHERE id NOT IN (
  SELECT DISTINCT user_id 
  FROM user_audit_logs 
  WHERE action = 'account_migrated' 
  AND user_id IS NOT NULL
);

-- Create a function to automatically log user changes
CREATE OR REPLACE FUNCTION log_user_changes()
RETURNS TRIGGER AS $$
BEGIN
  -- Log user updates (excluding password changes for security)
  IF TG_OP = 'UPDATE' AND OLD.* IS DISTINCT FROM NEW.* THEN
    INSERT INTO user_audit_logs (user_id, action, ip_address, user_agent, metadata)
    VALUES (
      NEW.id,
      'profile_updated',
      COALESCE(current_setting('app.client_ip', true), 'system'),
      COALESCE(current_setting('app.user_agent', true), 'system'),
      json_build_object(
        'changed_fields', (
          SELECT array_agg(key)
          FROM (
            SELECT key
            FROM jsonb_each_text(to_jsonb(OLD)) old_data
            FULL OUTER JOIN jsonb_each_text(to_jsonb(NEW)) new_data USING (key)
            WHERE old_data.value IS DISTINCT FROM new_data.value
            AND key NOT IN ('password_hash', 'updated_at', 'password_reset_token', 'email_verification_token')
          ) changed
        ),
        'timestamp', NOW()
      )
    );
  END IF;

  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger for user changes
DROP TRIGGER IF EXISTS trigger_log_user_changes ON users;
CREATE TRIGGER trigger_log_user_changes
  AFTER UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION log_user_changes();

-- Create a function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  DELETE FROM sessions WHERE expires_at < NOW();
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  
  -- Log cleanup activity
  INSERT INTO user_audit_logs (user_id, action, ip_address, user_agent, metadata)
  VALUES (
    NULL,
    'session_cleanup',
    'system',
    'system',
    json_build_object(
      'deleted_sessions', deleted_count,
      'cleanup_timestamp', NOW()
    )
  );
  
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create a function to clean up old audit logs (keep last 6 months)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs()
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  DELETE FROM user_audit_logs 
  WHERE created_at < NOW() - INTERVAL '6 months'
  AND action NOT IN ('login_success', 'login_failed', 'account_migrated');
  
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  
  -- Log cleanup activity
  INSERT INTO user_audit_logs (user_id, action, ip_address, user_agent, metadata)
  VALUES (
    NULL,
    'audit_cleanup',
    'system',
    'system',
    json_build_object(
      'deleted_logs', deleted_count,
      'cleanup_timestamp', NOW()
    )
  );
  
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create a view for active user sessions
CREATE OR REPLACE VIEW active_user_sessions AS
SELECT 
  s.id as session_id,
  s.user_id,
  u.email,
  u.first_name,
  u.last_name,
  u.role,
  s.ip_address,
  s.user_agent,
  s.created_at as session_started,
  s.expires_at as session_expires,
  EXTRACT(EPOCH FROM (s.expires_at - NOW()))::INTEGER as seconds_until_expiry
FROM sessions s
JOIN users u ON s.user_id = u.id
WHERE s.expires_at > NOW()
ORDER BY s.created_at DESC;

-- Create a view for user security summary
CREATE OR REPLACE VIEW user_security_summary AS
SELECT 
  u.id,
  u.email,
  u.role,
  u.is_active,
  u.email_verified IS NOT NULL as email_verified,
  u.two_factor_enabled,
  u.last_login_at,
  u.login_attempts,
  u.lockout_until,
  (
    SELECT COUNT(*) 
    FROM sessions s 
    WHERE s.user_id = u.id AND s.expires_at > NOW()
  ) as active_sessions_count,
  (
    SELECT COUNT(*) 
    FROM user_audit_logs ual 
    WHERE ual.user_id = u.id 
    AND ual.action = 'login_failed' 
    AND ual.created_at > NOW() - INTERVAL '24 hours'
  ) as failed_logins_24h,
  (
    SELECT ual.created_at 
    FROM user_audit_logs ual 
    WHERE ual.user_id = u.id 
    AND ual.action = 'login_success'
    ORDER BY ual.created_at DESC 
    LIMIT 1
  ) as last_successful_login
FROM users u;

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON sessions TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_audit_logs TO postgres;
GRANT SELECT ON active_user_sessions TO postgres;
GRANT SELECT ON user_security_summary TO postgres;
GRANT EXECUTE ON FUNCTION cleanup_expired_sessions() TO postgres;
GRANT EXECUTE ON FUNCTION cleanup_old_audit_logs() TO postgres;

-- Add comments for documentation
COMMENT ON TABLE sessions IS 'User authentication sessions managed by Lucia v3';
COMMENT ON TABLE user_audit_logs IS 'Security audit trail for user actions and system events';
COMMENT ON VIEW active_user_sessions IS 'Currently active user sessions with expiry information';
COMMENT ON VIEW user_security_summary IS 'Comprehensive security overview for each user';
COMMENT ON FUNCTION cleanup_expired_sessions() IS 'Removes expired sessions and logs cleanup activity';
COMMENT ON FUNCTION cleanup_old_audit_logs() IS 'Archives old audit logs while preserving important security events';

-- Success message
DO $$
BEGIN
  RAISE NOTICE 'Enhanced authentication schema migration completed successfully';
  RAISE NOTICE 'Added security features: email verification, password reset, 2FA support, audit logging';
  RAISE NOTICE 'Created % active sessions view and % security summary view', 
    (SELECT COUNT(*) FROM active_user_sessions),
    (SELECT COUNT(*) FROM user_security_summary);
END $$;