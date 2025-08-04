-- Basic Legal AI schema without pgvector
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name TEXT,
    role VARCHAR(50) DEFAULT 'prosecutor' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_number VARCHAR(50) NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'open' NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium' NOT NULL,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    evidence_type VARCHAR(50) NOT NULL,
    file_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO users (email, name, role) VALUES 
('admin@legal-ai.com', 'Admin User', 'admin'),
('prosecutor@legal-ai.com', 'John Prosecutor', 'prosecutor')
ON CONFLICT (email) DO NOTHING;

INSERT INTO cases (case_number, title, description, status, priority, created_by) 
SELECT 'CASE-2024-001', 'Sample Legal Case', 'This is a test case for the legal AI system', 'open', 'high', u.id
FROM users u WHERE u.email = 'prosecutor@legal-ai.com'
ON CONFLICT (case_number) DO NOTHING;

INSERT INTO evidence (case_id, title, description, evidence_type, file_url)
SELECT c.id, 'Document Evidence', 'Sample legal document', 'document', '/uploads/sample.pdf'
FROM cases c WHERE c.case_number = 'CASE-2024-001'
ON CONFLICT DO NOTHING;

SELECT 'Legal AI database schema created successfully!' AS status;