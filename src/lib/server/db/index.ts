// Unified DB export surface
// Copied from sveltekit-frontend/src/lib/server/db/index.ts on 2025-08-31
// Provides a single source of truth for `db` and related exports.
// Central DB export surface
import { drizzle } from 'drizzle-orm/postgres-js';
// Re-exported helpers are exported directly from 'drizzle-orm' below
import postgres from 'postgres';

import * as schema from './schema-postgres';

// NOTE: schema-postgres is often a generated or runtime-only module and may not export
// named symbols at build time; avoid re-exporting everything from it here to prevent
// TypeScript errors when those symbols are not present.
// Keep the schema object imported above for runtime use by drizzle and export shared helpers.
export { sql, eq, and, or, count, like, ilike, isNull, isNotNull, ne, gte, lte, desc, asc } from 'drizzle-orm';
export type { SQL } from 'drizzle-orm';

// Database connection
const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

const queryClient = postgres(connectionString);
export const db = drizzle(queryClient, { schema: schema as any });

const migrationClient = postgres(connectionString, { max: 1 });
export const migrationDb = drizzle(migrationClient, { schema: schema as any });

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

// Backwards-compatible alias: many files import getDatabaseHealth
export const getDatabaseHealth = healthCheck;

// CamelCase aliases for schema items removed because schema-postgres may be a runtime-only
// or generated module that does not expose these named symbols at build time.
// Consumers should import the generated schema module directly when they need specific
// table exports, or access runtime schema via the `schema` import above.
/* no direct re-exports from './schema-postgres' to avoid missing-symbol TypeScript errors */

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

// Database type detection function
export const isPostgreSQL = (): boolean => {
  const dbUrl = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';
  return dbUrl.startsWith('postgresql://') || dbUrl.startsWith('postgres://');
};