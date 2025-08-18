-- Create basic tables for YoRHa Legal AI Platform
-- Simplified schema for immediate testing

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name TEXT,
    role VARCHAR(50) DEFAULT 'prosecutor' NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Cases table
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number VARCHAR(50) NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    priority VARCHAR(20) DEFAULT 'medium' NOT NULL,
    status VARCHAR(20) DEFAULT 'open' NOT NULL,
    category VARCHAR(50),
    danger_score INTEGER DEFAULT 0 NOT NULL,
    jurisdiction VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Legal documents table
CREATE TABLE IF NOT EXISTS legal_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    jurisdiction VARCHAR(100),
    court VARCHAR(200),
    citation VARCHAR(300),
    content TEXT,
    summary TEXT,
    keywords JSONB DEFAULT '[]',
    topics JSONB DEFAULT '[]',
    case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Evidence table
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    evidence_type VARCHAR(50) NOT NULL,
    file_type VARCHAR(50),
    collected_by VARCHAR(255),
    location TEXT,
    summary TEXT,
    is_admissible BOOLEAN DEFAULT true NOT NULL,
    confidentiality_level VARCHAR(20) DEFAULT 'standard' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Insert seed data
INSERT INTO users (email, name, role) VALUES 
    ('detective@yorha.ai', 'Detective Adams', 'detective'),
    ('prosecutor@yorha.ai', 'Prosecutor Williams', 'prosecutor'),
    ('admin@yorha.ai', 'YoRHa Administrator', 'admin')
ON CONFLICT (email) DO NOTHING;

INSERT INTO cases (case_number, title, description, priority, status, category, danger_score, jurisdiction) VALUES 
    ('YORHA-2024-001', 'Corporate Fraud Investigation', 'High-profile corporate fraud case involving financial manipulation and insider trading.', 'high', 'active', 'white_collar_crime', 85, 'Federal District Court'),
    ('YORHA-2024-002', 'Employment Discrimination Lawsuit', 'Class action lawsuit regarding systemic employment discrimination practices.', 'medium', 'pending', 'civil_rights', 45, 'State Superior Court'),
    ('YORHA-2024-003', 'Intellectual Property Theft', 'Technology company accused of stealing trade secrets and patent infringement.', 'high', 'active', 'intellectual_property', 70, 'Federal Circuit Court')
ON CONFLICT (case_number) DO NOTHING;

INSERT INTO legal_documents (title, document_type, jurisdiction, court, citation, content, summary, keywords, topics) VALUES 
    ('Securities Exchange Act Violation Analysis', 'case_law', 'Federal', 'District Court for SDNY', '15 U.S.C. § 78j(b)', 'Comprehensive analysis of securities fraud under Rule 10b-5, examining materiality standards, scienter requirements, and reliance elements in corporate disclosure cases.', 'Analysis of Rule 10b-5 violations in corporate disclosure requirements.', '["securities fraud", "Rule 10b-5", "materiality", "scienter"]', '["corporate law", "securities regulation", "fraud"]'),
    ('Title VII Employment Discrimination Precedent', 'statute', 'Federal', 'Supreme Court', '42 U.S.C. § 2000e', 'Supreme Court precedent establishing the framework for analyzing employment discrimination claims under Title VII, including burden-shifting mechanisms and proof standards.', 'Key precedent on disparate treatment and disparate impact under Title VII.', '["Title VII", "discrimination", "disparate treatment", "burden shifting"]', '["employment law", "civil rights", "discrimination"]'),
    ('Trade Secret Protection Under DTSA', 'regulation', 'Federal', 'Federal Circuit', '18 U.S.C. § 1836', 'Comprehensive overview of the Defend Trade Secrets Act, including definition of trade secrets, misappropriation standards, and available remedies for intellectual property theft.', 'Defense Trade Secrets Act enforcement and remedies analysis.', '["trade secrets", "DTSA", "misappropriation", "injunctive relief"]', '["intellectual property", "trade secrets", "federal law"]');

INSERT INTO evidence (title, description, evidence_type, file_type, collected_by, location, summary, is_admissible, confidentiality_level) VALUES 
    ('Financial Records - Q3 2024', 'Quarterly financial statements showing discrepancies in revenue reporting.', 'document', 'pdf', 'Detective Adams', 'Corporate Headquarters - Finance Department', 'Critical financial evidence showing pattern of revenue manipulation and false reporting to SEC.', true, 'restricted'),
    ('Email Chain - Executive Communications', 'Internal executive email communications discussing discriminatory hiring practices.', 'digital', 'email', 'Prosecutor Williams', 'Corporate Email Server', 'Email evidence demonstrating intentional discriminatory practices in hiring and promotion decisions.', true, 'confidential'),
    ('Source Code Repository', 'Stolen proprietary algorithms and trade secret documentation.', 'digital', 'source_code', 'Detective Adams', 'Defendant Company Servers', 'Digital forensics evidence showing unauthorized access and theft of proprietary technology.', true, 'top_secret');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
CREATE INDEX IF NOT EXISTS idx_legal_documents_type ON legal_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(evidence_type);
CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_admin;