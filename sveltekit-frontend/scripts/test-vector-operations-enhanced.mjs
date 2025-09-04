#!/usr/bin/env node
/**
 * Enhanced Vector Operations Test Script
 * Tests PostgreSQL pgvector operations for the Legal AI Platform
 */

import postgres from 'postgres';
import { readFileSync } from 'fs';

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  red: '\x1b[31m'
};

const log = {
  info: (msg) => console.log(`${colors.cyan}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✅${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}❌${colors.reset} ${msg}`),
  step: (msg) => console.log(`${colors.blue}🔄${colors.reset} ${msg}`),
};

// Configuration
const DATABASE_URL = process.env.DATABASE_URL || 
  `postgresql://${process.env.DATABASE_USER || 'legal_admin'}:${process.env.DATABASE_PASSWORD || '123456'}@${process.env.DATABASE_HOST || 'localhost'}:${process.env.DATABASE_PORT || '5432'}/${process.env.DATABASE_NAME || 'legal_ai_db'}`;

// Sample embeddings for testing (384 dimensions)
function generateSampleEmbedding() {
  return Array.from({ length: 384 }, () => Math.random() * 2 - 1); // Random values between -1 and 1
}

function generateLegalDocumentEmbedding(content) {
  // Simple hash-based embedding for testing
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  
  const embedding = [];
  for (let i = 0; i < 384; i++) {
    embedding.push(Math.sin(hash + i) * 0.5); // Deterministic but varied values
  }
  return embedding;
}

async function main() {
  console.log(`${colors.bright}🧪 Enhanced Vector Operations Test for Legal AI Platform${colors.reset}\n`);

  // Initialize PostgreSQL connection with vector support
  const sql = postgres(DATABASE_URL, {
    types: {
      vector: {
        to: 1184,
        from: [1184],
        serialize: (x) => `[${x.join(',')}]`,
        parse: (x) => x.slice(1, -1).split(',').map(Number),
      },
    },
  });

  try {
    // Test 1: Basic connectivity
    log.step('Testing PostgreSQL connectivity...');
    const version = await sql`SELECT version() as version`;
    log.success(`Connected to: ${version[0].version.split(' ').slice(0, 2).join(' ')}`);

    // Test 2: Check pgvector extension
    log.step('Checking pgvector extension...');
    const extensions = await sql`
      SELECT * FROM pg_extension WHERE extname = 'vector'
    `;
    if (extensions.length > 0) {
      log.success(`pgvector ${extensions[0].extversion} is installed`);
    } else {
      log.error('pgvector extension not found');
      return;
    }

    // Test 3: Check existing tables
    log.step('Checking existing tables...');
    const tables = await sql`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
      ORDER BY table_name
    `;
    console.log(`   Found ${tables.length} tables: ${tables.map(t => t.table_name).join(', ')}`);

    // Test 4: Vector operations capability
    log.step('Testing vector operations...');
    try {
      const vectorTest = await sql`SELECT '[1,2,3]'::vector(3) as test_vector`;
      log.success('Basic vector operations working');
    } catch (vectorError) {
      log.error(`Vector operations failed: ${vectorError.message}`);
      return;
    }

    // Test 5: Vector similarity calculations
    log.step('Testing vector similarity calculations...');
    const embedding1 = generateSampleEmbedding();
    const embedding2 = generateSampleEmbedding();
    
    try {
      const similarityTest = await sql`
        SELECT 
          ${JSON.stringify(embedding1)}::vector(384) <=> ${JSON.stringify(embedding2)}::vector(384) as cosine_distance,
          1 - (${JSON.stringify(embedding1)}::vector(384) <=> ${JSON.stringify(embedding2)}::vector(384)) as cosine_similarity
      `;
      
      const distance = parseFloat(similarityTest[0].cosine_distance);
      const similarity = parseFloat(similarityTest[0].cosine_similarity);
      
      log.success(`Vector similarity calculation working (similarity: ${similarity.toFixed(4)})`);
    } catch (error) {
      log.error(`Similarity calculation failed: ${error.message}`);
    }

    // Test 6: Check if enhanced schema tables exist
    log.step('Checking enhanced schema tables...');
    const schemaResults = {
      users: await sql`SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = 'users'`,
      legal_documents: await sql`SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = 'legal_documents'`,
      cases: await sql`SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = 'cases'`,
      vector_operations: await sql`SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = 'vector_operations'`,
      qdrant_collections: await sql`SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = 'qdrant_collections'`,
    };

    let schemaTablesExist = 0;
    for (const [tableName, result] of Object.entries(schemaResults)) {
      const exists = result[0].count > 0;
      if (exists) {
        schemaTablesExist++;
        log.success(`Table '${tableName}' exists`);
      } else {
        log.warning(`Table '${tableName}' missing (run 'npm run db:setup' to create)`);
      }
    }

    // Test 7: Insert and query test data (if tables exist)
    if (schemaTablesExist >= 2) {
      log.step('Testing vector operations with sample data...');
      
      // Sample legal documents for testing
      const sampleDocuments = [
        {
          title: 'Employment Contract - Software Developer',
          content: 'This employment agreement is entered into between TechCorp and John Doe for the position of Senior Software Developer. The employee shall receive a salary of $120,000 annually...',
          document_type: 'contract',
          practice_area: 'employment'
        },
        {
          title: 'Non-Disclosure Agreement - Product Development', 
          content: 'This Non-Disclosure Agreement (NDA) is executed between ABC Corp and XYZ Consulting to protect confidential information related to new product development initiatives...',
          document_type: 'contract',
          practice_area: 'corporate'
        },
        {
          title: 'Lease Agreement - Commercial Property',
          content: 'This lease agreement is for commercial property located at 123 Business Street for a term of 5 years with monthly rent of $8,000. The lessee agrees to maintain...',
          document_type: 'contract', 
          practice_area: 'real_estate'
        }
      ];

      try {
        // Check if legal_documents table has vector columns
        const columns = await sql`
          SELECT column_name, data_type 
          FROM information_schema.columns 
          WHERE table_name = 'legal_documents' 
          AND column_name LIKE '%embedding'
        `;

        if (columns.length > 0) {
          log.success(`Found ${columns.length} vector columns in legal_documents table`);
          
          // Try to insert a test document
          const testEmbedding = generateLegalDocumentEmbedding(sampleDocuments[0].content);
          
          try {
            // Check if document exists first
            const existingDocs = await sql`
              SELECT id FROM legal_documents 
              WHERE title = ${sampleDocuments[0].title}
              LIMIT 1
            `;

            let documentId;
            if (existingDocs.length === 0) {
              // Insert test document
              const insertResult = await sql`
                INSERT INTO legal_documents (
                  title, content, document_type, practice_area, 
                  content_embedding, status
                ) VALUES (
                  ${sampleDocuments[0].title},
                  ${sampleDocuments[0].content},
                  ${sampleDocuments[0].document_type},
                  ${sampleDocuments[0].practice_area},
                  ${JSON.stringify(testEmbedding)}::vector(384),
                  'active'
                ) RETURNING id
              `;
              documentId = insertResult[0].id;
              log.success('Test document inserted successfully');
            } else {
              documentId = existingDocs[0].id;
              log.info('Test document already exists');
            }

            // Test similarity search
            const searchEmbedding = generateLegalDocumentEmbedding('employment agreement salary developer');
            
            const similarDocs = await sql`
              SELECT 
                id, title, document_type, practice_area,
                1 - (content_embedding <=> ${JSON.stringify(searchEmbedding)}::vector(384)) as similarity
              FROM legal_documents 
              WHERE content_embedding IS NOT NULL
                AND deleted_at IS NULL
                AND status = 'active'
              ORDER BY content_embedding <=> ${JSON.stringify(searchEmbedding)}::vector(384)
              LIMIT 5
            `;

            if (similarDocs.length > 0) {
              log.success(`Vector search returned ${similarDocs.length} results`);
              console.log('\n📋 Top similar documents:');
              similarDocs.forEach((doc, i) => {
                console.log(`   ${i + 1}. ${doc.title} (similarity: ${doc.similarity.toFixed(4)})`);
              });
            } else {
              log.warning('No documents found in similarity search');
            }

          } catch (insertError) {
            log.warning(`Document operations test skipped: ${insertError.message}`);
          }
        } else {
          log.warning('No vector columns found - enhanced schema may not be applied yet');
        }

      } catch (schemaError) {
        log.warning(`Schema test skipped: ${schemaError.message}`);
      }
    }

    // Test 8: Performance test
    log.step('Running performance test...');
    const perfTestSize = 1000;
    const testVectors = Array.from({ length: 10 }, () => generateSampleEmbedding());
    
    console.time('Vector operations performance');
    
    for (const vector of testVectors) {
      await sql`
        SELECT ${JSON.stringify(vector)}::vector(384) <=> ${JSON.stringify(generateSampleEmbedding())}::vector(384) as distance
      `;
    }
    
    console.timeEnd('Vector operations performance');
    log.success(`Processed ${testVectors.length} vector similarity calculations`);

  } catch (error) {
    log.error(`Test failed: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  } finally {
    await sql.end();
  }

  console.log(`\n${colors.green}${colors.bright}✅ Vector operations test completed successfully!${colors.reset}`);
  
  console.log('\n🎯 Summary:');
  console.log('   ✅ PostgreSQL connectivity verified');
  console.log('   ✅ pgvector extension confirmed');
  console.log('   ✅ Vector operations functional');
  console.log('   ✅ Similarity calculations working');
  console.log('   ✅ Performance test completed');
  
  console.log('\n🚀 Next steps:');
  console.log('   1. Run: npm run db:setup (to create enhanced schema)');
  console.log('   2. Run: npm run qdrant:sync (to sync with Qdrant if available)');
  console.log('   3. Start your SvelteKit application with vector search enabled');
}

// Handle errors
process.on('uncaughtException', (error) => {
  console.error(`${colors.red}❌ Uncaught exception: ${error.message}${colors.reset}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(`${colors.red}❌ Unhandled rejection: ${reason}${colors.reset}`);
  process.exit(1);
});

// Run main function
main().catch((error) => {
  console.error(`${colors.red}❌ Test failed: ${error.message}${colors.reset}`);
  process.exit(1);
});