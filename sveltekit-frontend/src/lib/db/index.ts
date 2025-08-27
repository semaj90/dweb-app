/**
 * Database Connection Module
 * Centralized database access point for the Legal AI Platform
 */
export { drizzle } from '$lib/server/db/index.js';
export type { Database } from '$lib/server/db/index.js';

// Re-export commonly used database utilities
export { 
  eq, 
  and, 
  or, 
  sql,
  desc,
  asc,
  count,
  isNull,
  isNotNull,
  like,
  ilike
} from 'drizzle-orm';

// Re-export schema types and tables
export * from '$lib/server/db/schema-postgres-enhanced.js';

// Default export for convenience
export { drizzle as db } from '$lib/server/db/index.js';