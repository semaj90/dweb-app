/**
 * Database Connection with pgvector Support
 */

import { drizzle } from 'drizzle-orm/node-postgres';
import postgres from 'postgres';
import * as schema from './schema-unified-postgres';

// Connection configuration
const connectionString = process.env.DATABASE_URL || 
  `postgresql://${process.env.POSTGRES_USER || 'legal_admin'}:${process.env.POSTGRES_PASSWORD || '123456'}@${process.env.POSTGRES_HOST || 'localhost'}:${process.env.POSTGRES_PORT || '5432'}/${process.env.POSTGRES_DB || 'legal_ai_db'}`;

// Create postgres connection with optimized settings
const sql = postgres(connectionString, {
  max: 20,
  idle_timeout: 20,
  connect_timeout: 10,
  // Enable pgvector extension
  onnotice: () => {}, // Suppress notices
  prepare: true, // Prepare statements for better performance
});

// Create Drizzle instance with schema
export const db = drizzle(sql, { schema });

// Test connection and ensure pgvector is installed
export async function initializeDatabase(): Promise<any> {
  try {
    console.log('🔌 Connecting to PostgreSQL...');
    
    // Test basic connection
    await sql`SELECT 1`;
    console.log('✅ Database connection successful');
    
    // Check for pgvector extension
    const [pgvectorCheck] = await sql`
      SELECT EXISTS(
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
      ) as installed
    `;
    
    if (pgvectorCheck.installed) {
      console.log('✅ pgvector extension is installed');
    } else {
      console.log('⚠️ pgvector extension not found, attempting to install...');
      try {
        await sql`CREATE EXTENSION IF NOT EXISTS vector`;
        console.log('✅ pgvector extension installed successfully');
      } catch (error: any) {
        console.error('❌ Failed to install pgvector extension:', error);
      }
    }
    
    // Check database schema version
    const [migration] = await sql`
      SELECT EXISTS(
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = '__drizzle_migrations__'
      ) as exists
    `;
    
    if (migration.exists) {
      const migrations = await sql`
        SELECT id, hash, created_at 
        FROM __drizzle_migrations__ 
        ORDER BY created_at DESC 
        LIMIT 1
      `;
      console.log('📊 Database migration status:', migrations.length > 0 ? 'Up to date' : 'Needs migration');
    } else {
      console.log('⚠️ No migration table found - run migrations first');
    }
    
    return true;
  } catch (error: any) {
    console.error('❌ Database connection failed:', error);
    return false;
  }
}

// Utility functions for vector operations
export async function performSimilaritySearch(
  embedding: number[],
  table: 'evidence' | 'chat_messages' | 'document_embeddings',
  column: string = 'embedding',
  limit: number = 10,
  threshold: number = 0.7
): Promise<any> {
  const embeddingStr = `[${embedding.join(',')}]`;
  
  const results = await sql`
    SELECT *, 
           1 - (${sql(column)} <=> ${embeddingStr}::vector) AS similarity
    FROM ${sql(table)}
    WHERE 1 - (${sql(column)} <=> ${embeddingStr}::vector) > ${threshold}
    ORDER BY ${sql(column)} <=> ${embeddingStr}::vector
    LIMIT ${limit}
  `;
  
  return results;
}

// Cleanup function
export async function closeConnection(): Promise<any> {
  await sql.end();
}

// Export connection for direct SQL queries
export { sql };