#!/usr/bin/env node

import pg from 'pg';

const client = new pg.Client({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

async function seedExistingDatabase() {
  try {
    await client.connect();
    console.log('✅ Connected to existing YoRHa database');

    // Insert legal documents using the correct schema
    console.log('📄 Seeding legal_documents table...');
    await client.query(`
      INSERT INTO legal_documents (filename, title, full_text, document_type, processing_status, created_at, updated_at) VALUES
      ('smith-v-jones-contract.pdf', 'Smith v. Jones Contract Dispute', 'Contract dispute regarding liability limitations and damages. The court found that liability clauses must be clearly defined and cannot unfairly limit reasonable damages. This case establishes precedent for contract interpretation in commercial disputes.', 'case_law', 'completed', NOW(), NOW()),
      ('employment-law-statute-2024.pdf', 'Employment Law Statute 2024', 'Employment discrimination protection statute establishing new standards for workplace equality. The statute provides comprehensive protections against discrimination based on race, gender, religion, and sexual orientation.', 'statute', 'completed', NOW(), NOW()),
      ('corporate-liability-precedent.pdf', 'Corporate Liability Precedent', 'Supreme Court ruling on corporate liability standards. This landmark decision clarifies the extent of corporate responsibility for executive actions and establishes new standards for piercing the corporate veil.', 'precedent', 'completed', NOW(), NOW())
      ON CONFLICT DO NOTHING;
    `);

    // Insert cases
    console.log('⚖️ Seeding cases table...');
    await client.query(`
      INSERT INTO cases (case_number, title, description, status, priority, danger_score, created_at, updated_at) VALUES
      ('YORHA-2024-001', 'Corporate Fraud Investigation', 'High-profile corporate fraud case involving financial manipulation and insider trading. Multiple executives implicated in securities violations.', 'active', 'high', 85, NOW(), NOW()),
      ('YORHA-2024-002', 'Employment Discrimination', 'Class action lawsuit regarding discriminatory hiring practices. Pattern of bias documented across multiple departments.', 'pending', 'medium', 45, NOW(), NOW()),
      ('YORHA-2024-003', 'IP Theft Investigation', 'Technology company accused of stealing trade secrets and patent infringement. Digital forensics evidence shows unauthorized access.', 'active', 'high', 70, NOW(), NOW())
      ON CONFLICT (case_number) DO NOTHING;
    `);

    // Insert evidence
    console.log('🔍 Seeding evidence table...');
    await client.query(`
      INSERT INTO evidence (title, description, evidence_type, collected_by, summary, is_admissible, created_at, updated_at) VALUES
      ('Email Chain Evidence', 'Executive email communications showing intent to manipulate financial reports', 'digital', 'Detective Adams', 'Critical evidence showing conspiracy to commit securities fraud', true, NOW(), NOW()),
      ('Contract Documents', 'Original signed contracts with discriminatory clauses', 'document', 'Prosecutor Williams', 'Legal documents proving intentional discrimination in hiring practices', true, NOW(), NOW()),
      ('Source Code Repository', 'Stolen proprietary algorithms and trade secret documentation', 'digital', 'Detective Adams', 'Digital forensics evidence showing unauthorized access and theft', true, NOW(), NOW())
    `);

    console.log('✅ Database seeding completed successfully!');

    // Verify the data
    const docCount = await client.query('SELECT COUNT(*) FROM legal_documents');
    const caseCount = await client.query('SELECT COUNT(*) FROM cases');
    const evidenceCount = await client.query('SELECT COUNT(*) FROM evidence');

    console.log('\n📊 YoRHa Database Status:');
    console.log(`   📄 Legal Documents: ${docCount.rows[0].count}`);
    console.log(`   ⚖️ Cases: ${caseCount.rows[0].count}`);
    console.log(`   🔍 Evidence: ${evidenceCount.rows[0].count}`);

    // Test a sample query
    console.log('\n🧪 Testing sample queries...');
    const sampleDocs = await client.query('SELECT title, document_type FROM legal_documents LIMIT 3');
    console.log('📄 Sample Legal Documents:');
    sampleDocs.rows.forEach(doc => {
      console.log(`   - ${doc.title} (${doc.document_type})`);
    });

    const sampleCases = await client.query('SELECT title, status, priority FROM cases LIMIT 3');
    console.log('⚖️ Sample Cases:');
    sampleCases.rows.forEach(case_ => {
      console.log(`   - ${case_.title} (${case_.status}, ${case_.priority})`);
    });

    console.log('\n🎯 YoRHa Legal AI Platform database is ready!');

  } catch (error) {
    console.error('❌ Database seeding failed:', error);
  } finally {
    await client.end();
  }
}

seedExistingDatabase();