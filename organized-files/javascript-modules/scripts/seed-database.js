const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

async function seedDatabase() {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL || 'postgresql://legal_admin:LegalRAG20245432/legal_rag_db'
  });

  try {
    console.log('🌱 Seeding legal database...');

    // Insert sample legal documents
    await pool.query(`
      INSERT INTO legal_documents (id, title, content, document_type, case_id) VALUES
      (gen_random_uuid(), 'Sample Contract - Software License', 'This software license agreement...', 'contract', 'RAG-2024-001'),
      (gen_random_uuid(), 'Employment Agreement Template', 'This employment agreement sets forth...', 'contract', 'RAG-2024-002'),
      (gen_random_uuid(), 'Case Brief - Contract Dispute', 'In the matter of TechCorp vs StartupXYZ...', 'case_brief', 'RAG-2024-001')
      ON CONFLICT DO NOTHING
    `);

    console.log('✅ Database seeded successfully');
    process.exit(0);

  } catch (error) {
    console.error('❌ Database seeding failed:', error);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

seedDatabase();
