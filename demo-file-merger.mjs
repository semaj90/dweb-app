// Simplified Legal AI File Merger Demo
import { S3Client, CreateBucketCommand, PutObjectCommand } from '@aws-sdk/client-s3';
import { Client } from 'pg';
import fs from 'fs';
import path from 'path';

console.log('🚀 Legal AI File Merger Demo Starting...');

// Configuration
const s3Client = new S3Client({
  endpoint: 'http://localhost:9000',
  region: 'us-east-1',
  credentials: {
    accessKeyId: 'minioadmin',
    secretAccessKey: 'minioadmin',
  },
  forcePathStyle: true,
});

const pgClient = new Client({
  connectionString: 'postgresql://postgres:123456@localhost:5432/legal_ai_db'
});

async function demo() {
  try {
    // Connect to PostgreSQL
    console.log('🔗 Connecting to PostgreSQL...');
    await pgClient.connect();

    // Create bucket
    console.log('🪣 Creating legal-documents bucket...');
    try {
      await s3Client.send(new CreateBucketCommand({ Bucket: 'legal-documents' }));
      console.log('✅ Bucket created successfully');
    } catch (error) {
      if (error.name === 'BucketAlreadyOwnedByYou') {
        console.log('✅ Bucket already exists');
      } else {
        throw error;
      }
    }

    // Create table
    console.log('📋 Creating legal_documents table...');
    await pgClient.query(`
      CREATE TABLE IF NOT EXISTS legal_documents (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        s3_bucket VARCHAR(100),
        s3_key TEXT,
        file_size BIGINT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        document_type VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);
    console.log('✅ Table ready');

    // Find a sample PDF
    const lawpdfsDir = './lawpdfs';
    const files = fs.readdirSync(lawpdfsDir).filter(f => f.endsWith('.pdf'));

    if (files.length === 0) {
      console.log('❌ No PDF files found in ./lawpdfs');
      return;
    }

    const sampleFile = files[0];
    const filePath = path.join(lawpdfsDir, sampleFile);
    const stats = fs.statSync(filePath);

    console.log(`📄 Processing sample file: ${sampleFile} (${(stats.size / 1024).toFixed(1)} KB)`);

    // Upload to S3
    const s3Key = `legal-docs/demo/${sampleFile}`;
    const fileStream = fs.createReadStream(filePath);

    await s3Client.send(new PutObjectCommand({
      Bucket: 'legal-documents',
      Key: s3Key,
      Body: fileStream,
      ContentType: 'application/pdf'
    }));

    console.log(`✅ Uploaded to S3: s3://legal-documents/${s3Key}`);

    // Store metadata in PostgreSQL
    const result = await pgClient.query(`
      INSERT INTO legal_documents (filename, s3_bucket, s3_key, file_size, document_type)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING id;
    `, [sampleFile, 'legal-documents', s3Key, stats.size, 'legal_document']);

    const docId = result.rows[0].id;
    console.log(`✅ Stored in PostgreSQL with ID: ${docId}`);

    // Verify the complete setup
    const count = await pgClient.query('SELECT COUNT(*) FROM legal_documents');
    console.log(`📊 Total documents in database: ${count.rows[0].count}`);

    console.log('\n🎉 Integration Demo Complete!');
    console.log('📍 Access points:');
    console.log('   🖥️  MinIO Console: http://localhost:9001 (minioadmin/minioadmin)');
    console.log('   📊 PostgreSQL: legal_ai_db.legal_documents table');
    console.log('   🔍 S3 API: http://localhost:9000');

  } catch (error) {
    console.error('❌ Demo failed:', error.message);
  } finally {
    await pgClient.end();
  }
}

demo();
