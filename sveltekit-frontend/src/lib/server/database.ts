// Real database connection configuration
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from "postgres";
import { pgTable, serial, text, timestamp, jsonb, vector, real, uuid } from 'drizzle-orm/pg-core';

// Database connection
const connectionString = process.env.DATABASE_URL || 
  `postgresql://${process.env.POSTGRES_USER || 'legal_admin'}:${process.env.POSTGRES_PASSWORD || '123456'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'legal_ai_db'}`;

const sql = postgres(connectionString, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
});

export const db = drizzle(sql);

// Database schemas
export const documents = pgTable('documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  filename: text('filename').notNull(),
  content: text('content').notNull(),
  originalContent: text('original_content'),
  metadata: jsonb('metadata'),
  confidence: real('confidence'),
  legalAnalysis: jsonb('legal_analysis'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});

export const embeddings = pgTable('legal_embeddings', {
  id: uuid('id').primaryKey().defaultRandom(),
  documentId: uuid('document_id').references(() => documents.id),
  content: text('content').notNull(),
  embedding: vector('embedding', { dimensions: 768 }),
  metadata: jsonb('metadata'),
  model: text('model').default('nomic-embed-text'),
  createdAt: timestamp('created_at').defaultNow(),
});

export const searchSessions = pgTable('search_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  query: text('query').notNull(),
  queryEmbedding: vector('query_embedding', { dimensions: 768 }),
  results: jsonb('results'),
  searchType: text('search_type').default('hybrid'),
  resultCount: serial('result_count'),
  createdAt: timestamp('created_at').defaultNow(),
});

// Initialize database with extensions
export async function initializeDatabase() {
  try {
    console.log('[Database] Initializing database...');
    
    // Create vector extension
    await sql`CREATE EXTENSION IF NOT EXISTS vector`;
    
    // Create full-text search extension
    await sql`CREATE EXTENSION IF NOT EXISTS pg_trgm`;
    
    // Create index for vector similarity search
    await sql`
      CREATE INDEX IF NOT EXISTS legal_embeddings_embedding_idx 
      ON legal_embeddings USING ivfflat (embedding vector_cosine_ops) 
      WITH (lists = 100)
    `;
    
    // Create full-text search index
    await sql`
      CREATE INDEX IF NOT EXISTS documents_content_fts_idx 
      ON documents USING gin(to_tsvector('english', content))
    `;
    
    // Create metadata indexes
    await sql`
      CREATE INDEX IF NOT EXISTS documents_metadata_idx 
      ON documents USING gin(metadata)
    `;
    
    console.log('[Database] Database initialized successfully');
    return true;
    
  } catch (error) {
    console.error('[Database] Initialization failed:', error);
    return false;
  }
}

// Test database connection
export async function testDatabaseConnection() {
  try {
    const result = await sql`SELECT 1 as test`;
    return result.length > 0;
  } catch (error) {
    console.error('[Database] Connection test failed:', error);
    return false;
  }
}
