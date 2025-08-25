// Central DB export surface - re-export canonical schema and selected auth artifacts
export * from './schema-unified';

import * as schema from "./schema-unified";
// Database connection and schema exports
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { sql, eq, and, or, desc, asc, count, like, ilike, isNull, isNotNull, ne, SQL } from 'drizzle-orm';

// Re-export sql and common query helpers for convenience across server code
export { sql, eq, and, or, desc, asc, count, like, ilike, isNull, isNotNull, ne };
// Export SQL type for utilities that reference it
export type { SQL };

// Database type helper - exported first to avoid temporal dead zone
export const isPostgreSQL = true;

// Use the schema directly
export const fullSchema = schema;

// Create the connection
const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

// For query purposes
const queryClient = postgres(connectionString);
export const db = drizzle(queryClient, { schema: fullSchema });

// For migrations
const migrationClient = postgres(connectionString, { max: 1 });
export const migrationDb = drizzle(migrationClient);

// Note: we intentionally re-export only the canonical schema module(s).
// Avoid exporting both `schema-postgres` and `schema-unified` because
// they contain overlapping symbol names which causes duplicate-export errors.

// Helper function to test database connection
export async function testConnection() {
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
  } catch (error) {
    console.error('❌ Database connection failed:', error);
    return false;
  }
}

// Initialize pgvector on first run
if (process.env.NODE_ENV !== 'production') {
  testConnection().catch(console.error);
}

// Health check function for API routes
export async function healthCheck() {
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
  } catch (error) {
    return {
      status: 'unhealthy',
      database: 'disconnected',
      error: error.message,
      tablesAccessible: false
    };
  }
}