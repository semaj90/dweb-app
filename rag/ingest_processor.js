// rag/ingest_processor.js - Enhanced ingestion with MinIO, PostgreSQL+PGVector, and nomic-embed
import fs from 'fs';
import Redis from 'ioredis';
import { Client as MinioClient } from 'minio';
import { Client as PgClient } from 'pg';

const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
const QUEUE_MODE = process.env.RAG_QUEUE_MODE || 'list';
const STREAM_KEY = 'rag_ingest_jobs_stream';
const GROUP = 'rag_proc_group';

// PostgreSQL configuration
const POSTGRES_DSN =
  process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db';
const DOCS_TABLE = 'documents';
const EMBEDDINGS_TABLE = 'legal_embeddings';
const DIM = 768; // nomic-embed-text dimensions
const BATCH = Number(process.env.RAG_EMBED_BATCH || 8);

// MinIO configuration
const minioClient = new MinioClient({
  endPoint: process.env.MINIO_ENDPOINT || 'localhost',
  port: parseInt(process.env.MINIO_PORT || '9000'),
  useSSL: process.env.MINIO_USE_SSL === 'true',
  accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin',
});

// Ollama configuration for nomic-embed
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const EMBEDDING_MODEL = 'nomic-embed-text';

let pg = null;
let pgAvailable = true;
try {
  pg = new PgClient({ connectionString: POSTGRES_DSN });
  await pg.connect();
  console.log('✅ PostgreSQL connected');
} catch (e) {
  pgAvailable = false;
  console.warn('[ingest-processor] PG connect failed:', e.message);
}

// Generate embeddings using Ollama nomic-embed
async function generateEmbeddings(textChunks) {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        prompt: Array.isArray(textChunks) ? textChunks.join('\n') : textChunks,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.error('[ingest-processor] Embedding generation failed:', error);
    throw error;
  }
}

// Store document in MinIO
async function storeInMinIO(bucketName, fileName, content) {
  try {
    await minioClient.putObject(bucketName, fileName, content);
    console.log(`✅ Document stored in MinIO: ${fileName}`);
    return true;
  } catch (error) {
    console.error('[ingest-processor] MinIO storage failed:', error);
    return false;
  }
}

// Ensure vector database tables exist
async function ensureVectorDatabase() {
  if (!pgAvailable) return false;

  try {
    // Create vector extension if not exists
    await pg.query('CREATE EXTENSION IF NOT EXISTS vector');

    // Create documents table
    await pg.query(`
      CREATE TABLE IF NOT EXISTS ${DOCS_TABLE} (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        filename VARCHAR(255) NOT NULL,
        content_type VARCHAR(100),
        file_size BIGINT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB DEFAULT '{}',
        status VARCHAR(50) DEFAULT 'pending',
        minio_object_name VARCHAR(255)
      )
    `);

    // Create embeddings table with vector column
    await pg.query(`
      CREATE TABLE IF NOT EXISTS ${EMBEDDINGS_TABLE} (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID REFERENCES ${DOCS_TABLE}(id) ON DELETE CASCADE,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER DEFAULT 0,
        embedding VECTOR(${DIM}),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create vector similarity index
    await pg.query(`
      CREATE INDEX IF NOT EXISTS idx_legal_embeddings_vector
      ON ${EMBEDDINGS_TABLE} USING ivfflat (embedding vector_cosine_ops)
      WITH (lists = 100)
    `);

    console.log('✅ Vector database tables ensured');
    return true;
  } catch (error) {
    console.error('[ingest-processor] Database setup failed:', error);
    return false;
  }
}

async function ensureSchema() {
  await ensureVectorDatabase();
}
await ensureSchema();

// Initialize MinIO bucket
async function ensureMinIOBucket() {
  try {
    const bucketName = 'legal-documents';
    const bucketExists = await minioClient.bucketExists(bucketName);
    if (!bucketExists) {
      await minioClient.makeBucket(bucketName);
      console.log(`✅ MinIO bucket created: ${bucketName}`);
    } else {
      console.log(`✅ MinIO bucket exists: ${bucketName}`);
    }
  } catch (error) {
    console.error('[ingest-processor] MinIO bucket setup failed:', error);
  }
}
await ensureMinIOBucket();

// Enhanced text chunking with overlap
function chunkText(text, maxChunkSize = 2000, overlap = 200) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + maxChunkSize, text.length);
    chunks.push(text.substring(start, end));
    start = end - overlap;
    if (start >= text.length) break;
  }
  return chunks;
}

// Enhanced job processing with MinIO and PostgreSQL integration
async function processJob(job) {
  if (!job?.text_path && !job?.content) return;

  let raw;
  if (job.text_path && fs.existsSync(job.text_path)) {
    raw = fs.readFileSync(job.text_path, 'utf8');
  } else if (job.content) {
    raw = job.content;
  } else {
    console.warn('[ingest-processor] No valid content found for job');
    return;
  }

  const fileName = job.filename || job.doc_id || `document_${Date.now()}.txt`;
  const chunks = chunkText(raw);

  if (!pgAvailable) {
    console.warn('[ingest-processor] PostgreSQL not available, skipping job');
    return;
  }

  try {
    // Store document metadata in PostgreSQL
    const docResult = await pg.query(
      `
      INSERT INTO ${DOCS_TABLE} (filename, content_type, file_size, metadata, status, minio_object_name)
      VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING id
    `,
      [
        fileName,
        job.content_type || 'text/plain',
        raw.length,
        JSON.stringify(job.metadata || {}),
        'processing',
        `${fileName}_${Date.now()}`,
      ]
    );

    const documentId = docResult.rows[0].id;

    // Store full document in MinIO
    const minioObjectName = `${fileName}_${Date.now()}`;
    await storeInMinIO('legal-documents', minioObjectName, raw);

    // Process chunks in batches
    for (let i = 0; i < chunks.length; i += BATCH) {
      const slice = chunks.slice(i, i + BATCH);

      // Generate embeddings for each chunk
      for (let j = 0; j < slice.length; j++) {
        try {
          const embedding = await generateEmbeddings(slice[j]);

          // Store chunk and embedding in PostgreSQL
          await pg.query(
            `
            INSERT INTO ${EMBEDDINGS_TABLE} (document_id, chunk_text, chunk_index, embedding)
            VALUES ($1, $2, $3, $4)
          `,
            [
              documentId,
              slice[j],
              i + j,
              `[${embedding.join(',')}]`, // PostgreSQL vector format
            ]
          );
        } catch (error) {
          console.error(`[ingest-processor] Failed to process chunk ${i + j}:`, error);
        }
      }
    }

    // Update document status to completed
    await pg.query(
      `
      UPDATE ${DOCS_TABLE}
      SET status = 'completed', minio_object_name = $1
      WHERE id = $2
    `,
      [minioObjectName, documentId]
    );

    console.log(
      `✅ [ingest-processor] Successfully indexed document: ${fileName} (${chunks.length} chunks)`
    );
  } catch (error) {
    console.error('[ingest-processor] Job processing failed:', error);

    // Update document status to failed if it was created
    if (documentId) {
      await pg
        .query(
          `
        UPDATE ${DOCS_TABLE}
        SET status = 'failed'
        WHERE id = $1
      `,
          [documentId]
        )
        .catch(() => {}); // Ignore update errors
    }
  }
}

async function ensureStream() {
  if (QUEUE_MODE !== 'stream') return;
  try {
    await redis.xgroup('CREATE', STREAM_KEY, GROUP, '$', 'MKSTREAM');
  } catch (e) {
    if (!/BUSYGROUP/.test(e.message))
      console.warn('[ingest-processor] stream group create failed:', e.message);
  }
}
await ensureStream();

async function loopList() {
  const raw = await redis.lpop('rag_ingest_jobs');
  if (raw) {
    try {
      await processJob(JSON.parse(raw));
    } catch (e) {
      console.warn('[ingest-processor] job error', e.message);
    }
  } else await new Promise((r) => setTimeout(r, 1200));
}

async function loopStream() {
  try {
    const res = await redis.xreadgroup(
      'GROUP',
      GROUP,
      'c1',
      'COUNT',
      5,
      'BLOCK',
      2000,
      'STREAMS',
      STREAM_KEY,
      '>'
    );
    if (Array.isArray(res)) {
      for (const [, entries] of res) {
        for (const [id, fields] of entries) {
          const idx = fields.findIndex((v, i) => i % 2 === 0 && v === 'job');
          const payload = idx >= 0 ? fields[idx + 1] : null;
          if (payload) {
            try {
              await processJob(JSON.parse(payload));
            } catch (e) {
              console.warn('stream job error', e.message);
            }
          }
          await redis.xack(STREAM_KEY, GROUP, id).catch(() => {});
        }
      }
    }
  } catch (e) {
    if (!/NOGROUP/.test(e.message)) console.warn('[ingest-processor] stream read error', e.message);
    await new Promise((r) => setTimeout(r, 500));
  }
}

async function mainLoop() {
  while (true) {
    if (QUEUE_MODE === 'stream') await loopStream();
    else await loopList();
  }
}
mainLoop();
