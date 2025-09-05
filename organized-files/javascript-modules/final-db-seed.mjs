#!/usr/bin/env node

import pg from 'pg';

const client = new pg.Client({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

async function finalDatabaseSeed() {
  try {
    await client.connect();
    console.log('✅ Connected to YoRHa Legal AI Database');

    // Insert legal documents using correct schema
    console.log('📄 Seeding legal_documents...');
    await client.query(`
      INSERT INTO legal_documents (filename, title, full_text, document_type, processing_status, created_at, updated_at) VALUES
      ('smith-v-jones-contract.pdf', 'Smith v. Jones Contract Dispute', 'Contract dispute regarding liability limitations and damages. The court found that liability clauses must be clearly defined and cannot unfairly limit reasonable damages. This case establishes precedent for contract interpretation in commercial disputes involving force majeure clauses and limitation of liability provisions.', 'case_law', 'completed', NOW(), NOW()),
      ('employment-statute-2024.pdf', 'Employment Law Statute 2024', 'Employment discrimination protection statute establishing new standards for workplace equality. The statute provides comprehensive protections against discrimination based on race, gender, religion, and sexual orientation. Key provisions include mandatory reporting procedures and enhanced penalties for violations.', 'statute', 'completed', NOW(), NOW()),
      ('corporate-liability-precedent.pdf', 'Corporate Liability Precedent', 'Supreme Court ruling on corporate liability standards. This landmark decision clarifies the extent of corporate responsibility for executive actions and establishes new standards for piercing the corporate veil in cases of fraud and misconduct.', 'precedent', 'completed', NOW(), NOW())
      ON CONFLICT DO NOTHING;
    `);

    // Insert cases using correct schema  
    console.log('⚖️ Seeding cases...');
    await client.query(`
      INSERT INTO cases (case_number, title, description, status, priority, created_at, updated_at) VALUES
      ('YORHA-2024-001', 'Corporate Fraud Investigation', 'High-profile corporate fraud case involving financial manipulation and insider trading. Multiple executives implicated in securities violations with evidence of systematic accounting fraud.', 'active', 'high', NOW(), NOW()),
      ('YORHA-2024-002', 'Employment Discrimination Lawsuit', 'Class action lawsuit regarding discriminatory hiring practices. Pattern of bias documented across multiple departments with statistical evidence of disparate impact.', 'pending', 'medium', NOW(), NOW()),
      ('YORHA-2024-003', 'Intellectual Property Theft', 'Technology company accused of stealing trade secrets and patent infringement. Digital forensics evidence shows unauthorized access to proprietary systems.', 'active', 'high', NOW(), NOW())
      ON CONFLICT (case_number) DO NOTHING;
    `);

    // Get case IDs for evidence linking
    const caseIds = await client.query('SELECT id, case_number FROM cases WHERE case_number LIKE \'YORHA-2024-%\'');
    const case1Id = caseIds.rows.find(c => c.case_number === 'YORHA-2024-001')?.id;
    const case2Id = caseIds.rows.find(c => c.case_number === 'YORHA-2024-002')?.id;
    const case3Id = caseIds.rows.find(c => c.case_number === 'YORHA-2024-003')?.id;

    // Insert evidence using correct schema
    console.log('🔍 Seeding evidence...');
    if (case1Id && case2Id && case3Id) {
      await client.query(`
        INSERT INTO evidence (case_id, title, description, evidence_type, created_at, updated_at) VALUES
        ($1, 'Financial Records Q3 2024', 'Quarterly financial statements showing discrepancies in revenue reporting and potential securities fraud', 'document', NOW(), NOW()),
        ($2, 'Email Chain - Executive Communications', 'Internal executive email communications discussing discriminatory hiring practices and bias in promotion decisions', 'digital', NOW(), NOW()),
        ($3, 'Source Code Repository', 'Stolen proprietary algorithms and trade secret documentation obtained through unauthorized network access', 'digital', NOW(), NOW())
      `, [case1Id, case2Id, case3Id]);
    }

    console.log('✅ YoRHa database seeding completed successfully!');

    // Verify the data
    const docCount = await client.query('SELECT COUNT(*) FROM legal_documents');
    const caseCount = await client.query('SELECT COUNT(*) FROM cases');
    const evidenceCount = await client.query('SELECT COUNT(*) FROM evidence');

    console.log('\n📊 YoRHa Legal AI Database Status:');
    console.log(`   📄 Legal Documents: ${docCount.rows[0].count}`);
    console.log(`   ⚖️ Cases: ${caseCount.rows[0].count}`);
    console.log(`   🔍 Evidence: ${evidenceCount.rows[0].count}`);

    // Test sample queries for YoRHa API
    console.log('\n🧪 Testing YoRHa API data...');
    
    const sampleDocs = await client.query(`
      SELECT title, document_type, processing_status 
      FROM legal_documents 
      WHERE processing_status = 'completed' 
      LIMIT 3
    `);
    console.log('📄 Legal Documents (for YoRHa Documents tab):');
    sampleDocs.rows.forEach(doc => {
      console.log(`   - ${doc.title} (${doc.document_type})`);
    });

    const sampleCases = await client.query(`
      SELECT case_number, title, status, priority 
      FROM cases 
      WHERE case_number LIKE 'YORHA-%' 
      LIMIT 3
    `);
    console.log('\n⚖️ Cases (for YoRHa Cases tab):');
    sampleCases.rows.forEach(case_ => {
      console.log(`   - ${case_.case_number}: ${case_.title} (${case_.status}, ${case_.priority})`);
    });

    const sampleEvidence = await client.query(`
      SELECT e.title, e.evidence_type, c.case_number 
      FROM evidence e 
      JOIN cases c ON e.case_id = c.id 
      LIMIT 3
    `);
    console.log('\n🔍 Evidence (for YoRHa Evidence tab):');
    sampleEvidence.rows.forEach(ev => {
      console.log(`   - ${ev.title} (${ev.evidence_type}) - Case: ${ev.case_number}`);
    });

    console.log('\n🎯 YoRHa Legal AI Platform is ready for testing!');
    console.log('🌐 Access the dashboard at: http://localhost:5173/yorha-dashboard');

  } catch (error) {
    console.error('❌ Database seeding failed:', error);
  } finally {
    await client.end();
  }
}

finalDatabaseSeed();