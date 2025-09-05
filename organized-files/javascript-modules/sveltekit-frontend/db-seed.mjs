#!/usr/bin/env node

import { drizzle } from 'drizzle-orm/node-postgres';
import pkg from 'pg';
const { Pool } = pkg;
import { hash } from '@node-rs/argon2';

// Database connection - try different user credentials
const dbConfigs = [
  'postgresql://postgres:123456@localhost:5432/legal_ai_db',
  'postgresql://postgres:postgres@localhost:5432/legal_ai_db', 
  'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db'
];

let pool;
let db;

async function connectToDatabase() {
  for (const config of dbConfigs) {
    try {
      pool = new Pool({ connectionString: config });
      db = drizzle(pool);
      
      // Test the connection
      await pool.query('SELECT 1 as test');
      console.log('✅ Database connection successful with:', config.replace(/:[^:@]*@/, ':****@'));
      return true;
    } catch (error) {
      console.log(`❌ Failed to connect with: ${config.replace(/:[^:@]*@/, ':****@')}`);
    }
  }
  throw new Error('Could not connect to database with any credentials');
}

// Generate sample vector embeddings (768 dimensions for Nomic embed-text)
function generateSampleEmbedding() {
  return Array.from({ length: 768 }, () => Math.random() * 2 - 1); // Random values between -1 and 1
}

async function seed() {
  console.log('🌱 Starting database seed...');

  try {
    // Connect to database
    await connectToDatabase();

    // Check if pgvector extension is installed
    try {
      const extensionCheck = await pool.query(`
        SELECT * FROM pg_extension WHERE extname = 'vector';
      `);
      if (extensionCheck.rows.length === 0) {
        console.log('⚠️  pgvector extension not found, attempting to create...');
        await pool.query(`CREATE EXTENSION IF NOT EXISTS vector;`);
        console.log('✅ pgvector extension created');
      } else {
        console.log('✅ pgvector extension already installed');
      }
    } catch (error) {
      console.log('⚠️  pgvector extension check failed:', error.message);
    }

    // Create password hash
    const passwordHash = await hash('password123', {
      memoryCost: 19456,
      timeCost: 2,
      outputLen: 32,
      parallelism: 1,
    });

    // Seed users
    console.log('👤 Creating users...');
    const seedUsers = [
      {
        email: 'prosecutor@legal.ai',
        name: 'John Prosecutor',
        firstName: 'John',
        lastName: 'Prosecutor',
        role: 'prosecutor',
        hashedPassword: passwordHash,
        isActive: true,
      },
      {
        email: 'detective@legal.ai', 
        name: 'Jane Detective',
        firstName: 'Jane',
        lastName: 'Detective',
        role: 'detective',
        hashedPassword: passwordHash,
        isActive: true,
      },
      {
        email: 'admin@legal.ai',
        name: 'Admin User', 
        firstName: 'Admin',
        lastName: 'User',
        role: 'admin',
        hashedPassword: passwordHash,
        isActive: true,
      }
    ];

    // Insert users with conflict handling
    const insertedUsers = [];
    for (const user of seedUsers) {
      try {
        const result = await db.insert(users).values(user).returning().execute();
        insertedUsers.push(result[0]);
        console.log(`✅ Created user: ${user.email}`);
      } catch (error) {
        if (error.message?.includes('unique')) {
          console.log(`ℹ️  User ${user.email} already exists`);
          // Get existing user
          const existing = await db.select().from(users).where(eq(users.email, user.email)).limit(1);
          if (existing[0]) insertedUsers.push(existing[0]);
        } else {
          throw error;
        }
      }
    }

    // Seed cases
    console.log('📁 Creating cases...');
    const seedCases = [
      {
        caseNumber: 'CASE-2024-001',
        title: 'Financial Fraud Investigation',
        description: 'Complex financial fraud case involving multiple entities and cryptocurrency transactions. Suspected money laundering through shell companies with international wire transfers exceeding $2.3M.',
        priority: 'high',
        status: 'open',
        category: 'financial_fraud',
        dangerScore: 75,
        createdBy: insertedUsers[0]?.id,
        aiSummary: 'High-priority financial fraud case with strong evidence of money laundering. Key suspects identified through transaction pattern analysis.',
        aiTags: ['money_laundering', 'cryptocurrency', 'international', 'high_value'],
      },
      {
        caseNumber: 'CASE-2024-002',
        title: 'Cybercrime Investigation',
        description: 'Data breach and identity theft case with international connections. Over 100,000 personal records compromised including SSNs, credit card numbers, and personal identifying information.',
        priority: 'medium',
        status: 'open',
        category: 'cybercrime', 
        dangerScore: 60,
        createdBy: insertedUsers[1]?.id,
        aiSummary: 'Large-scale data breach affecting consumers across multiple states. Evidence suggests sophisticated APT group involvement.',
        aiTags: ['data_breach', 'identity_theft', 'apt_group', 'consumer_harm'],
      },
      {
        caseNumber: 'CASE-2024-003',
        title: 'White Collar Crime',
        description: 'Corporate embezzlement case with extensive document evidence. CFO suspected of diverting company funds to personal accounts over 3-year period.',
        priority: 'high',
        status: 'open', 
        category: 'embezzlement',
        dangerScore: 45,
        createdBy: insertedUsers[0]?.id,
        aiSummary: 'Clear evidence of systematic embezzlement through fraudulent invoicing and payment redirection. Total loss estimated at $1.8M.',
        aiTags: ['embezzlement', 'white_collar', 'corporate_fraud', 'systematic'],
      }
    ];

    const insertedCases = [];
    for (const caseData of seedCases) {
      try {
        const result = await db.insert(cases).values(caseData).returning().execute();
        insertedCases.push(result[0]);
        console.log(`✅ Created case: ${caseData.caseNumber}`);
      } catch (error) {
        if (error.message?.includes('unique')) {
          console.log(`ℹ️  Case ${caseData.caseNumber} already exists`);
        } else {
          throw error;
        }
      }
    }

    // Seed evidence  
    console.log('🔍 Creating evidence...');
    if (insertedCases.length > 0) {
      const seedEvidence = [
        {
          caseId: insertedCases[0]?.id,
          title: 'Bank Transaction Records',
          description: 'Suspicious transaction patterns showing money laundering activity. Records span 18 months and include wire transfers to offshore accounts.',
          evidenceType: 'financial_document',
          tags: ['transactions', 'banking', 'offshore'],
          uploadedBy: insertedUsers[0]?.id,
          aiAnalysis: {
            confidence: 0.92,
            patterns: ['structuring', 'round_amounts', 'threshold_avoidance'],
            recommendations: ['Focus on accounts 4457 and 8821', 'Check international wire transfer compliance'],
          },
          aiTags: ['money_laundering', 'suspicious_patterns', 'high_confidence'],
          aiSummary: 'Strong evidence of money laundering through structured transactions. 15 suspicious transactions identified totaling $2.3M.',
        },
        {
          caseId: insertedCases[1]?.id,
          title: 'Server Logs', 
          description: 'Access logs showing unauthorized data access and exfiltration. Evidence of SQL injection attacks and privilege escalation.',
          evidenceType: 'digital_evidence',
          tags: ['logs', 'unauthorized_access', 'data_exfiltration'],
          uploadedBy: insertedUsers[1]?.id,
          aiAnalysis: {
            confidence: 0.94,
            techniques: ['sql_injection', 'privilege_escalation', 'data_exfiltration'],
            timeline: '2024-01-15 to 2024-03-20',
          },
          aiTags: ['cyberattack', 'technical_evidence', 'timeline_established'],
          aiSummary: 'Server logs provide clear technical evidence of unauthorized access and systematic data exfiltration over 2-month period.',
        }
      ];

      for (const evidenceData of seedEvidence) {
        try {
          const result = await db.insert(evidence).values(evidenceData).returning().execute();
          console.log(`✅ Created evidence: ${evidenceData.title}`);
        } catch (error) {
          console.log(`⚠️  Evidence creation failed: ${error.message}`);
        }
      }
    }

    // Seed legal documents with embeddings (if pgvector is available)
    console.log('📄 Creating legal documents with embeddings...');
    try {
      const sampleLegalDoc = {
        title: 'Money Laundering Statute Reference',
        documentType: 'statute',
        jurisdiction: 'federal',
        content: 'Federal money laundering statutes define the criminal offense of engaging in financial transactions with proceeds of unlawful activity...',
        fullText: 'Complete statutory text would be here with full legal definitions, penalties, and procedural requirements.',
        embedding: generateSampleEmbedding(), // 768-dimensional vector
        keywords: ['money_laundering', 'financial_crimes', 'federal_statute'],
        topics: ['financial_crimes', 'money_laundering', 'criminal_law'],
        createdBy: insertedUsers[0]?.id,
      };

      const result = await db.insert(legalDocuments).values(sampleLegalDoc).returning().execute();
      console.log(`✅ Created legal document with embedding: ${sampleLegalDoc.title}`);
    } catch (error) {
      console.log(`⚠️  Legal document creation failed: ${error.message}`);
      console.log('This is expected if pgvector extension is not properly installed');
    }

    console.log('\n🎉 Database seed completed successfully!');
    console.log(`
📊 Summary:
   👤 Users: ${insertedUsers.length}
   📁 Cases: ${insertedCases.length}
   🔍 Evidence: Created evidence items
   📄 Legal Documents: Attempted with pgvector support

🔐 Test Login Credentials:
   prosecutor@legal.ai / password123  
   detective@legal.ai / password123
   admin@legal.ai / password123

🔧 Next Steps:
   1. Run migrations: npm run db:migrate
   2. Test login at: http://localhost:5173/login
   3. Explore vector search capabilities
`);

  } catch (error) {
    console.error('❌ Seed failed:', error);
    console.error('Full error:', error.stack);
  } finally {
    await client.end();
  }
}

// Run seed if called directly
if (import.meta.url === new URL(process.argv[1], import.meta.url).href) {
  seed();
}

export { seed };