#!/usr/bin/env node

/**
 * Legal AI File Merger with MinIO S3 Integration
 *
 * Comprehensive file processing system for legal documents:
 * - Scans source directory for legal PDFs
 * - Extracts text content and metadata
 * - Uploads files to MinIO S3-compatible storage
 * - Stores metadata in PostgreSQL
 * - Prepares vector embeddings for search
 *
 * Usage: node legal-ai-file-merger.mjs --source=./lawpdfs --bucket=legal-documents --verbose
 */

import { S3Client, CreateBucketCommand, PutObjectCommand, HeadBucketCommand } from '@aws-sdk/client-s3';
import { Client } from 'pg';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { createReadStream } from 'fs';
import pdf from 'pdf-parse';
import mime from 'mime-types';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration from environment variables
const config = {
  aws: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'minioadmin',
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'minioadmin',
    endpoint: process.env.AWS_ENDPOINT_URL || 'http://localhost:9000',
    region: 'us-east-1', // MinIO default
    forcePathStyle: true // Required for MinIO
  },
  postgres: {
    connectionString: process.env.POSTGRES_CONNECTION_STRING || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
  }
};

// Parse command line arguments
function parseArgs() {
  const args = {};
  process.argv.slice(2).forEach(arg => {
    if (arg.startsWith('--')) {
      const [key, value] = arg.slice(2).split('=');
      args[key] = value || true;
    }
  });
  return args;
}

// Initialize S3 client
const s3Client = new S3Client({
  endpoint: config.aws.endpoint,
  region: config.aws.region,
  credentials: {
    accessKeyId: config.aws.accessKeyId,
    secretAccessKey: config.aws.secretAccessKey,
  },
  forcePathStyle: config.aws.forcePathStyle,
});

// Initialize PostgreSQL client
let pgClient;

async function initDatabase() {
  pgClient = new Client({ connectionString: config.postgres.connectionString });
  await pgClient.connect();

  // Create legal_documents table if it doesn't exist
  await pgClient.query(`
    CREATE TABLE IF NOT EXISTS legal_documents (
      id SERIAL PRIMARY KEY,
      filename VARCHAR(255) NOT NULL,
      original_path TEXT,
      s3_bucket VARCHAR(100),
      s3_key TEXT,
      file_size BIGINT,
      mime_type VARCHAR(100),
      upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      document_type VARCHAR(50),
      title TEXT,
      content_preview TEXT,
      full_text TEXT,
      metadata JSONB,
      embedding_vector vector(1536),
      processing_status VARCHAR(20) DEFAULT 'uploaded',
      error_message TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `);

  console.log('✅ Database initialized with legal_documents table');
}

async function ensureBucket(bucketName) {
  try {
    // Check if bucket exists
    await s3Client.send(new HeadBucketCommand({ Bucket: bucketName }));
    console.log(`✅ Bucket '${bucketName}' already exists`);
  } catch (error) {
    if (error.name === 'NotFound') {
      // Create bucket
      await s3Client.send(new CreateBucketCommand({ Bucket: bucketName }));
      console.log(`✅ Created bucket '${bucketName}'`);
    } else {
      throw error;
    }
  }
}

async function extractPdfText(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const data = await pdf(dataBuffer);
    return {
      text: data.text,
      pages: data.numpages,
      info: data.info || {}
    };
  } catch (error) {
    console.error(`❌ Error extracting PDF text from ${filePath}:`, error.message);
    return { text: '', pages: 0, info: {}, error: error.message };
  }
}

function generateMetadata(filename, pdfData, stats) {
  const metadata = {
    filename,
    fileSize: stats.size,
    pages: pdfData.pages,
    extractedAt: new Date().toISOString(),
    pdfInfo: pdfData.info,
    contentLength: pdfData.text.length,
    wordCount: pdfData.text.split(/\s+/).filter(word => word.length > 0).length
  };

  // Extract document type from filename patterns
  const filename_lower = filename.toLowerCase();
  if (filename_lower.includes('people v.') || filename_lower.includes('court')) {
    metadata.documentType = 'court_case';
  } else if (filename_lower.includes('bill text') || filename_lower.includes('sb-') || filename_lower.includes('ab-')) {
    metadata.documentType = 'legislation';
  } else if (filename_lower.includes('usdoj') || filename_lower.includes('department of justice')) {
    metadata.documentType = 'federal_case';
  } else if (filename_lower.includes('code') || filename_lower.includes('statute')) {
    metadata.documentType = 'legal_code';
  } else {
    metadata.documentType = 'legal_document';
  }

  // Extract title from filename (remove .pdf and clean up)
  metadata.title = filename.replace(/\.pdf$/i, '').replace(/_/g, ' ');

  return metadata;
}

async function uploadToS3(filePath, bucketName, s3Key) {
  const fileStream = createReadStream(filePath);
  const mimeType = mime.lookup(filePath) || 'application/octet-stream';

  const uploadParams = {
    Bucket: bucketName,
    Key: s3Key,
    Body: fileStream,
    ContentType: mimeType,
    Metadata: {
      originalPath: filePath,
      uploadedBy: 'legal-ai-file-merger',
      uploadedAt: new Date().toISOString()
    }
  };

  await s3Client.send(new PutObjectCommand(uploadParams));
  return { bucketName, s3Key, mimeType };
}

async function storeInDatabase(fileInfo, s3Info, metadata, pdfData) {
  const contentPreview = pdfData.text.substring(0, 500) + (pdfData.text.length > 500 ? '...' : '');

  const query = `
    INSERT INTO legal_documents (
      filename, original_path, s3_bucket, s3_key, file_size, mime_type,
      document_type, title, content_preview, full_text, metadata, processing_status
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    RETURNING id;
  `;

  const values = [
    fileInfo.filename,
    fileInfo.filePath,
    s3Info.bucketName,
    s3Info.s3Key,
    fileInfo.size,
    s3Info.mimeType,
    metadata.documentType,
    metadata.title,
    contentPreview,
    pdfData.text,
    JSON.stringify(metadata),
    pdfData.error ? 'error' : 'processed'
  ];

  const result = await pgClient.query(query, values);
  return result.rows[0].id;
}

async function processFile(filePath, bucketName, verbose = false) {
  try {
    const filename = path.basename(filePath);
    const stats = await fs.stat(filePath);

    if (verbose) {
      console.log(`📄 Processing: ${filename} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
    }

    // Extract PDF content
    const pdfData = await extractPdfText(filePath);
    const metadata = generateMetadata(filename, pdfData, stats);

    // Generate S3 key with organized structure
    const datePrefix = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
    const s3Key = `legal-docs/${datePrefix}/${metadata.documentType}/${filename}`;

    // Upload to S3
    const s3Info = await uploadToS3(filePath, bucketName, s3Key);

    // Store metadata in PostgreSQL
    const docId = await storeInDatabase(
      { filename, filePath, size: stats.size },
      s3Info,
      metadata,
      pdfData
    );

    if (verbose) {
      console.log(`✅ Processed ${filename} -> Document ID: ${docId}`);
      console.log(`   📁 S3: s3://${bucketName}/${s3Key}`);
      console.log(`   📊 Content: ${pdfData.text.length} chars, ${pdfData.pages} pages`);
      console.log(`   🏷️  Type: ${metadata.documentType}`);
    }

    return { success: true, docId, s3Key, metadata };

  } catch (error) {
    console.error(`❌ Error processing ${path.basename(filePath)}:`, error.message);
    return { success: false, error: error.message };
  }
}

async function scanDirectory(sourceDir) {
  const files = [];

  async function scanRecursively(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        await scanRecursively(fullPath);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.pdf')) {
        files.push(fullPath);
      }
    }
  }

  await scanRecursively(sourceDir);
  return files;
}

async function main() {
  const args = parseArgs();
  const sourceDir = args.source || './lawpdfs';
  const bucketName = args.bucket || 'legal-documents';
  const verbose = args.verbose || false;
  const dryRun = args['dry-run'] || false;

  console.log('🚀 Legal AI File Merger Starting...');
  console.log(`📂 Source Directory: ${sourceDir}`);
  console.log(`🪣 S3 Bucket: ${bucketName}`);
  console.log(`🔗 MinIO Endpoint: ${config.aws.endpoint}`);
  console.log(`🗄️  PostgreSQL: ${config.postgres.connectionString.replace(/:[^:]*@/, ':***@')}`);

  if (dryRun) {
    console.log('🧪 DRY RUN MODE - No actual uploads will be performed');
  }

  try {
    // Initialize services
    console.log('\n🔧 Initializing services...');
    await initDatabase();

    if (!dryRun) {
      await ensureBucket(bucketName);
    }

    // Scan for files
    console.log(`\n📁 Scanning ${sourceDir} for PDF files...`);
    const files = await scanDirectory(sourceDir);
    console.log(`📋 Found ${files.length} PDF files to process`);

    if (files.length === 0) {
      console.log('ℹ️  No PDF files found. Exiting.');
      return;
    }

    // Process files
    console.log('\n🔄 Processing files...');
    const results = {
      total: files.length,
      successful: 0,
      failed: 0,
      errors: []
    };

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      console.log(`\n[${i + 1}/${files.length}] Processing: ${path.basename(file)}`);

      if (dryRun) {
        console.log(`   🧪 DRY RUN: Would process ${file}`);
        results.successful++;
        continue;
      }

      const result = await processFile(file, bucketName, verbose);

      if (result.success) {
        results.successful++;
      } else {
        results.failed++;
        results.errors.push({ file: path.basename(file), error: result.error });
      }
    }

    // Summary
    console.log('\n📊 Processing Summary:');
    console.log(`✅ Successful: ${results.successful}`);
    console.log(`❌ Failed: ${results.failed}`);
    console.log(`📈 Success Rate: ${((results.successful / results.total) * 100).toFixed(1)}%`);

    if (results.failed > 0) {
      console.log('\n❌ Errors encountered:');
      results.errors.forEach(error => {
        console.log(`   • ${error.file}: ${error.error}`);
      });
    }

    if (!dryRun && results.successful > 0) {
      console.log('\n🎉 File processing completed successfully!');
      console.log(`📚 ${results.successful} legal documents are now available in:`);
      console.log(`   🪣 MinIO S3: http://localhost:9001 (minioadmin/minioadmin)`);
      console.log(`   🗄️  PostgreSQL: legal_ai_db.legal_documents table`);
      console.log(`\n🔍 Next steps:`);
      console.log(`   • Generate embeddings for vector search`);
      console.log(`   • Create search indexes for fast retrieval`);
      console.log(`   • Integrate with RAG system for AI-powered legal research`);
    }

  } catch (error) {
    console.error('💥 Fatal error:', error.message);
    if (verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  } finally {
    if (pgClient) {
      await pgClient.end();
    }
  }
}

// Handle process termination gracefully
process.on('SIGINT', async () => {
  console.log('\n🛑 Received interrupt signal, cleaning up...');
  if (pgClient) {
    await pgClient.end();
  }
  process.exit(0);
});

// Run the main function
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('💥 Unhandled error:', error);
    process.exit(1);
  });
}
