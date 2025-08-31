/**
 * Database Connection Module
 * Centralized database access point for the Legal AI Platform
 */
// Re-export commonly used database utilities
export { 
  eq, 
  and, 
  or, 
  sql,
  count,
  isNull,
  isNotNull,
  like,
  ilike
} from 'drizzle-orm';

// Re-export schema tables explicitly (avoiding conflicts)
export {
  cases,
  evidence,
  legal_documents,
  documentChunks,
  users,
  sessions
} from '$lib/server/db/schema-postgres';

// Re-export types only from server db
export type * from '$lib/server/db/index';

// Default export for convenience - db is already exported above