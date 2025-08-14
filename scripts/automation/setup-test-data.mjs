/**
 * Setup Test Data for Enhanced REST API
 * Populates database with sample legal documents and embeddings
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { documents, cases } from './src/lib/server/db/schema-postgres.js';

// Database connection
const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/legal_ai';
const client = postgres(connectionString);
const db = drizzle(client);

// Sample legal documents with embeddings
const sampleDocuments = [
  {
    id: 'doc_contract_001',
    title: 'Software License Agreement',
    extractedText: 'This software license agreement governs the use of proprietary software with specific liability limitations and indemnification clauses. The licensee agrees to comply with all terms and conditions outlined herein.',
    embeddings: Array.from({ length: 384 }, () => Math.random() * 2 - 1),
    metadata: {
      type: 'contract',
      keywords: ['liability', 'indemnification', 'license', 'software'],
      legalTopics: ['intellectual_property', 'contract_law'],
      riskLevel: 'medium'
    }
  },
  {
    id: 'doc_case_002',
    title: 'Smith v. Tech Corp - Negligence Case',
    extractedText: 'Plaintiff alleges negligence in software design causing business interruption and financial losses. The court must determine liability standards for software defects in commercial applications.',
    embeddings: Array.from({ length: 384 }, () => Math.random() * 2 - 1),
    metadata: {
      type: 'case_law',
      keywords: ['negligence', 'damages', 'software_defect', 'business_interruption'],
      legalTopics: ['tort_law', 'commercial_law'],
      riskLevel: 'high'
    }
  },
  {
    id: 'doc_regulation_003',
    title: 'Data Protection Compliance Guide',
    extractedText: 'Organizations processing personal data must implement appropriate technical and organizational measures to ensure data protection compliance. This includes data minimization, purpose limitation, and security safeguards.',
    embeddings: Array.from({ length: 384 }, () => Math.random() * 2 - 1),
    metadata: {
      type: 'regulation',
      keywords: ['data_protection', 'privacy', 'compliance', 'gdpr'],
      legalTopics: ['privacy_law', 'regulatory_compliance'],
      riskLevel: 'high'
    }
  },
  {
    id: 'doc_evidence_004',
    title: 'Digital Forensics Analysis Report',
    extractedText: 'Forensic examination of digital devices reveals communication patterns and data access logs. Chain of custody procedures have been followed to ensure admissibility of digital evidence.',
    embeddings: Array.from({ length: 384 }, () => Math.random() * 2 - 1),
    metadata: {
      type: 'evidence',
      keywords: ['digital_forensics', 'chain_of_custody', 'admissibility', 'communication'],
      legalTopics: ['evidence_law', 'criminal_procedure'],
      riskLevel: 'medium'
    }
  },
  {
    id: 'doc_motion_005',
    title: 'Motion for Summary Judgment',
    extractedText: 'Defendant moves for summary judgment arguing no genuine dispute of material fact exists. The motion addresses liability limitations and contractual defenses under applicable commercial law.',
    embeddings: Array.from({ length: 384 }, () => Math.random() * 2 - 1),
    metadata: {
      type: 'filing',
      keywords: ['summary_judgment', 'material_fact', 'liability', 'commercial_law'],
      legalTopics: ['civil_procedure', 'commercial_law'],
      riskLevel: 'medium'
    }
  }
];

// Sample cases
const sampleCases = [
  {
    id: 'case_001',
    title: 'Tech Corp Liability Case',
    description: 'Commercial software liability dispute involving negligence claims',
    status: 'active',
    priority: 'high',
    createdBy: 'test_user',
    metadata: {
      jurisdiction: 'federal',
      caseType: 'commercial',
      estimatedValue: 500000
    }
  },
  {
    id: 'case_002',
    title: 'Data Protection Violation',
    description: 'GDPR compliance investigation and potential regulatory action',
    status: 'active',
    priority: 'high',
    createdBy: 'test_user',
    metadata: {
      jurisdiction: 'EU',
      caseType: 'regulatory',
      estimatedValue: 250000
    }
  }
];

async function setupTestData() {
  console.log('🔧 Setting up test data for Enhanced REST API...');
  
  try {
    // Insert sample cases
    console.log('📁 Inserting sample cases...');
    for (const caseData of sampleCases) {
      try {
        await db.insert(cases).values(caseData).onConflictDoNothing();
        console.log(`✅ Inserted case: ${caseData.title}`);
      } catch (error) {
        console.log(`⚠️ Case already exists or error: ${caseData.title}`);
      }
    }
    
    // Insert sample documents
    console.log('\\n📄 Inserting sample documents...');
    for (const docData of sampleDocuments) {
      try {
        await db.insert(documents).values({
          id: docData.id,
          title: docData.title,
          extractedText: docData.extractedText,
          embeddings: docData.embeddings,
          metadata: docData.metadata,
          caseId: sampleCases[0].id, // Associate with first case
          filename: `${docData.id}.pdf`,
          mimeType: 'application/pdf',
          size: 1024 * 10, // 10KB
          uploadedBy: 'test_user'
        }).onConflictDoNothing();
        console.log(`✅ Inserted document: ${docData.title}`);
      } catch (error) {
        console.log(`⚠️ Document already exists or error: ${docData.title}`);
      }
    }
    
    // Verify data insertion
    console.log('\\n🔍 Verifying test data...');
    const caseCount = await db.select().from(cases);
    const docCount = await db.select().from(documents);
    
    console.log(`📊 Cases in database: ${caseCount.length}`);
    console.log(`📊 Documents in database: ${docCount.length}`);
    
    // Display sample document for verification
    console.log('\\n📋 Sample document structure:');
    if (docCount.length > 0) {
      const sampleDoc = docCount[0];
      console.log(`ID: ${sampleDoc.id}`);
      console.log(`Title: ${sampleDoc.title}`);
      console.log(`Embedding dimensions: ${sampleDoc.embeddings?.length || 'N/A'}`);
      console.log(`Metadata: ${JSON.stringify(sampleDoc.metadata, null, 2)}`);
    }
    
    console.log('\\n✅ Test data setup completed!');
    console.log('\\n🚀 You can now run the API tests:');
    console.log('   node test-api-simple.mjs');
    console.log('   node test-enhanced-rest-api.mjs');
    
  } catch (error) {
    console.error('❌ Error setting up test data:', error);
    throw error;
  } finally {
    await client.end();
  }
}

// Run setup
setupTestData().catch(console.error);