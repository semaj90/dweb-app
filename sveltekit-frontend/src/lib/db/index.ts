/**
 * Database Connection Module
 * Centralized database access point for the Legal AI Platform
 */
export { drizzle, db } from '$lib/server/db/index';
export type { Database } from '$lib/server/db/index';

// Re-export commonly used database utilities
export { 
  eq, 
  and, 
  or, 
  sql
} from 'drizzle-orm';
export {
  count,
  isNull,
  isNotNull,
  like,
  ilike
} from 'drizzle-orm';

// Re-export schema types and tables
export * from '$lib/server/db/schema-postgres-enhanced';

// Default export for convenience - db is already exported above