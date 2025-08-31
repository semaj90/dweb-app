// Central DB export surface - re-export canonical schema and selected auth artifacts
// Export everything from the canonical schema module (primary surface)
export * from './schema-postgres';

// Explicit named re-exports for compatibility with varied import styles across
// the codebase. Some files import specific symbols from
// "$lib/server/db/schema-postgres" or expect particular camelCase/snake_case
// aliases; exporting them here makes the central `db` index a reliable shim.
export {
  users,
  sessions,
  cases,
  evidence,
  legal_documents,
  documentChunks,
  embeddingCache,
  keys,
  userProfiles,
  reports,
  statutes,
  legalAnalysisSessions,
  userAiQueries,
  autoTags,
  caseDocuments,
  caseActivities,
  caseTimeline,
  caseScores,
  ragSessions,
  ragMessages,
  vectorMetadata,
  criminals,
  personsOfInterest,
  canvasStates,
  caseDocuments as case_documents,
  documentChunks as document_chunks,
  legal_documents as legalDocuments,
  personsOfInterest as persons_of_interest,
  embeddingCache as embedding_cache
} from './schema-postgres';

// Provide camelCase and common alias re-exports for modules that expect different
// naming conventions. This reduces the number of files that need editing.
export { users as Users, users as UsersTable } from './schema-postgres';
export { cases as Cases, cases as CaseTable } from './schema-postgres';
export { caseActivities as case_activities, caseActivities as CaseActivities } from './schema-postgres';
export { personsOfInterest as persons_of_interest_alias, personsOfInterest as PersonsOfInterest } from './schema-postgres';
export { legal_documents as legalDocuments_alias, legal_documents as Documents } from './schema-postgres';

import * as schema from "./schema-postgres";

// === INFERRED TABLE TYPES ===
// Clean TypeScript types derived from Drizzle table definitions
import type {
  users,
  sessions,
  cases,
  evidence,
  legal_documents,
  documentChunks,
  embeddingCache
} from './schema-postgres';

// Select types (for reading data from database)
export type SelectUser = typeof users.$inferSelect;
export type SelectSession = typeof sessions.$inferSelect;
export type SelectCase = typeof cases.$inferSelect;
export type SelectEvidence = typeof evidence.$inferSelect;
export type SelectLegalDocument = typeof legal_documents.$inferSelect;
export type SelectDocumentChunk = typeof documentChunks.$inferSelect;
export type SelectEmbeddingCache = typeof embeddingCache.$inferSelect;

// Insert types (for creating new database records)
export type InsertUser = typeof users.$inferInsert;
export type InsertSession = typeof sessions.$inferInsert;
export type InsertCase = typeof cases.$inferInsert;
export type InsertEvidence = typeof evidence.$inferInsert;
export type InsertLegalDocument = typeof legal_documents.$inferInsert;
export type InsertDocumentChunk = typeof documentChunks.$inferInsert;
export type InsertEmbeddingCache = typeof embeddingCache.$inferInsert;
// Database connection and schema exports
// Use the postgres-js adapter since this project uses the `postgres` (porsager) client
// (importing node-postgres here caused runtime errors like "client.query is not a function").
import { drizzle } from 'drizzle-orm/postgres-js';
import { sql, eq, and, or, count, like, ilike, isNull, isNotNull, ne, gte, lte } from 'drizzle-orm';
// Import ordering functions from specific submodule
import { desc, asc } from 'drizzle-orm/sql/expressions/select';

// Re-export sql and common query helpers for convenience across server code
export { sql, eq, and, or, count, like, ilike, isNull, isNotNull, ne, gte, lte, desc, asc };
// Export SQL type for utilities that reference it - use lowercase
export type SQL = typeof sql;

// Database type helper - exported first to avoid temporal dead zone
export const isPostgreSQL = true;

// Use the schema directly
export const fullSchema = schema;

// Create the connection
const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

// For query purposes - use ESM import
import postgres from 'postgres';
const queryClient = postgres(connectionString);
export const db = drizzle(queryClient, { schema: fullSchema });

// Export drizzle constructor for client-side imports
export { drizzle };
export type Database = typeof db;

// For migrations
const migrationClient = postgres(connectionString, { max: 1 });
export const migrationDb = drizzle(migrationClient);

// Note: we intentionally re-export only the canonical schema module(s).
// Avoid exporting both `schema-postgres` and `schema-unified` because
// they contain overlapping symbol names which causes duplicate-export errors.

// Helper function to test database connection
export async function testConnection(): Promise<any> {
  try {
    await queryClient`SELECT 1`;
    console.log('✅ Database connection successful');

    // Check for pgvector extension
    const result = await queryClient`
      SELECT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
      ) as has_vector
    `;

    if (result[0].has_vector) {
      console.log('✅ pgvector extension is installed');
    } else {
      console.log('⚠️  pgvector extension not found, installing...');
      await queryClient`CREATE EXTENSION IF NOT EXISTS vector`;
      console.log('✅ pgvector extension installed');
    }

    return true;
  } catch (error: any) {
    console.error('❌ Database connection failed:', error);
    return false;
  }
}

// Initialize pgvector on first run
if (process.env.NODE_ENV !== 'production') {
  testConnection().catch(console.error);
}

// Health check function for API routes
export async function healthCheck(): Promise<any> {
  try {
    await queryClient`SELECT 1`;

    // Check if tables are accessible
    const tables = await queryClient`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
      LIMIT 5
    `;

    return {
      status: 'healthy',
      database: 'connected',
      tablesAccessible: tables.length > 0
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      database: 'disconnected',
      error: error.message,
      tablesAccessible: false
    };
  }
}

// === DATABASE UTILITIES ===
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

// Query result wrapper interface
export interface QueryResult<T = any> {
  data: T[];
  count: number;
  error?: string;
  success: boolean;
}

// Database configuration interface
export interface DatabaseConfig {
  connectionString: string;
  maxConnections?: number;
  idleTimeout?: number;
  ssl?: boolean;
}

// Re-export Drizzle type utilities for external use
export type {
  InferSelectModel,
  InferInsertModel
} from 'drizzle-orm/table';