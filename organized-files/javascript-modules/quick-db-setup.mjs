#!/usr/bin/env node

import pg from 'pg';

console.log('🚀 Quick YoRHa Database Setup');

const client = new pg.Client({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

async function setupDatabase() {
  try {
    await client.connect();
    console.log('✅ Connected to database');

    // Create basic tables
    console.log('📊 Creating tables...');
    
    await client.query(`
      CREATE TABLE IF NOT EXISTS legal_documents (
        id SERIAL PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        document_type VARCHAR(50) NOT NULL DEFAULT 'general',
        jurisdiction VARCHAR(100),
        court VARCHAR(200),
        citation VARCHAR(300),
        content TEXT,
        summary TEXT,
        keywords TEXT DEFAULT '[]',
        topics TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      );
    `);

    await client.query(`
      CREATE TABLE IF NOT EXISTS cases (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        case_number VARCHAR(50) UNIQUE,
        status VARCHAR(20) DEFAULT 'active',
        priority VARCHAR(20) DEFAULT 'medium',
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      );
    `);

    await client.query(`
      CREATE TABLE IF NOT EXISTS evidence (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        evidence_type VARCHAR(50) NOT NULL DEFAULT 'document',
        case_id INTEGER REFERENCES cases(id),
        collected_by VARCHAR(255),
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      );
    `);

    console.log('✅ Tables created successfully');

    // Insert sample data
    console.log('📝 Inserting sample data...');

    await client.query(`
      INSERT INTO legal_documents (title, document_type, content, summary) VALUES
      ('Smith v. Jones Contract Dispute', 'case_law', 'Contract dispute regarding liability limitations and damages...', 'Contract liability case with precedential value'),
      ('Employment Law Statute 2024', 'statute', 'Employment discrimination protection statute...', 'State employment protection law'),
      ('Corporate Liability Precedent', 'precedent', 'Supreme Court ruling on corporate liability standards...', 'Key corporate liability precedent')
      ON CONFLICT DO NOTHING;
    `);

    await client.query(`
      INSERT INTO cases (title, description, case_number, status, priority) VALUES
      ('Corporate Fraud Investigation', 'High-profile corporate fraud case involving financial manipulation', 'CASE-2024-001', 'active', 'high'),
      ('Employment Discrimination', 'Class action lawsuit regarding discriminatory hiring practices', 'CASE-2024-002', 'pending', 'medium'),
      ('IP Theft Investigation', 'Technology company accused of stealing trade secrets', 'CASE-2024-003', 'active', 'high')
      ON CONFLICT (case_number) DO NOTHING;
    `);

    await client.query(`
      INSERT INTO evidence (title, description, evidence_type, case_id, collected_by, status) VALUES
      ('Email Chain Evidence', 'Executive email communications showing intent', 'digital', 1, 'Detective Adams', 'verified'),
      ('Contract Documents', 'Original signed contracts with liability clauses', 'document', 2, 'Prosecutor Williams', 'processing'),
      ('Source Code Repository', 'Stolen proprietary algorithms and documentation', 'digital', 3, 'Detective Adams', 'verified')
      ON CONFLICT DO NOTHING;
    `);

    console.log('✅ Sample data inserted');
    console.log('🎯 YoRHa database setup complete!');

    // Test queries
    const docCount = await client.query('SELECT COUNT(*) FROM legal_documents');
    const caseCount = await client.query('SELECT COUNT(*) FROM cases');
    const evidenceCount = await client.query('SELECT COUNT(*) FROM evidence');

    console.log('\n📊 Database Status:');
    console.log(`   📄 Legal Documents: ${docCount.rows[0].count}`);
    console.log(`   ⚖️ Cases: ${caseCount.rows[0].count}`);
    console.log(`   🔍 Evidence: ${evidenceCount.rows[0].count}`);

  } catch (error) {
    console.error('❌ Database setup failed:', error);
  } finally {
    await client.end();
  }
}

setupDatabase();