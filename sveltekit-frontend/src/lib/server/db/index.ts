// Central DB export surface
import { drizzle } from 'drizzle-orm/postgres-js';
import { sql, eq, and, or, count, like, ilike, isNull, isNotNull, ne, gte, lte, desc } from 'drizzle-orm';
import { asc } from 'drizzle-orm/sql';
import postgres from 'postgres';

import * as schema from './schema-postgres';

// Re-export all schema components and types
export * from './schema-postgres';

// Re-export sql and common query helpers
export { sql, eq, and, or, count, like, ilike, isNull, isNotNull, ne, gte, lte, desc, asc };
export type { SQL } from 'drizzle-orm/sql';

// Database connection
const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

const queryClient = postgres(connectionString);
export const db = drizzle(queryClient, { schema });

const migrationClient = postgres(connectionString, { max: 1 });
export const migrationDb = drizzle(migrationClient, { schema });

export type Database = typeof db;

// Health check function
export async function healthCheck() {
  try {
    await queryClient`SELECT 1`;
    const result = await queryClient`SELECT extname FROM pg_extension WHERE extname = 'vector'`;
    const pgvector = result.length > 0;
    return {
      status: 'healthy',
      database: 'connected',
      pgvector,
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      database: 'disconnected',
      error: error.message,
    };
  }
}

// Commonly used enum types and constants
export type UserRole = 'user' | 'prosecutor' | 'investigator' | 'admin' | 'attorney' | 'paralegal';
export type CaseStatus = 'open' | 'active' | 'under_review' | 'closed' | 'archived' | 'draft';
export type EvidenceType = 'document' | 'image' | 'video' | 'audio' | 'physical' | 'digital';
export type DocumentType = 'evidence' | 'case_file' | 'report' | 'transcript' | 'correspondence' | 'legal_brief';

// Table name constants for dynamic queries
export const TABLE_NAMES = {
  USERS: 'users',
  SESSIONS: 'sessions',
  CASES: 'cases',
  EVIDENCE: 'evidence',
  LEGAL_DOCUMENTS: 'legal_documents',
  DOCUMENT_CHUNKS: 'document_chunks',
  EMBEDDING_CACHE: 'embedding_cache'
} as const;

export type TableName = typeof TABLE_NAMES[keyof typeof TABLE_NAMES];