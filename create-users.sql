DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'prosecutor' NOT NULL,
    department VARCHAR(100),
    badge_number VARCHAR(50),
    jurisdiction VARCHAR(100),
    display_name VARCHAR(255),
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true NOT NULL,
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW() NOT NULL
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    bio TEXT,
    phone VARCHAR(20),
    address TEXT,
    preferences JSONB DEFAULT '{}' NOT NULL,
    permissions JSONB DEFAULT '[]' NOT NULL,
    specializations JSONB DEFAULT '[]' NOT NULL,
    certifications JSONB DEFAULT '[]' NOT NULL,
    experience_level VARCHAR(20) DEFAULT 'junior',
    work_patterns JSONB DEFAULT '{}' NOT NULL,
    metadata JSONB DEFAULT '{}' NOT NULL,
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW() NOT NULL
);