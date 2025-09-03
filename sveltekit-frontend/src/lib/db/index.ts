/**
 * Database Connection with Drizzle ORM + pgvector Integration
 * PostgreSQL + pgvector support for YoRHa Legal AI Platform
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema.js';
import type { PostgresJsDatabase } from 'drizzle-orm/postgres-js';

// Environment configuration with fallbacks for legal_ai_db
const DATABASE_URL = import.meta.env.DATABASE_URL || 
  `postgresql://${import.meta.env.POSTGRES_USER || 'postgres'}:${import.meta.env.POSTGRES_PASSWORD || '123456'}@${import.meta.env.POSTGRES_HOST || 'localhost'}:${import.meta.env.POSTGRES_PORT || '5432'}/${import.meta.env.POSTGRES_DB || 'legal_ai_db'}`;

// Create postgres client with enhanced configuration for vector operations
const client = postgres(DATABASE_URL, {
  // Connection pool settings
  max: 20,
  idle_timeout: 20,
  connect_timeout: 10,
  
  // PostgreSQL-specific optimizations
  prepare: false, // Disable prepared statements for better compatibility
  
  // Enable vector extension support
  types: {
    // Custom type parser for vector data
    vector: {
      to: 1184,
      from: [1184],
      serialize: (value: number[]) => `[${value.join(',')}]`,
      parse: (value: string) => {
        const matches = value.match(/^\[(.*)\]$/);
        if (!matches) return [];
        return matches[1] ? matches[1].split(',').map(Number) : [];
      }
    }
  },
  
  // SSL configuration for production
  ssl: import.meta.env.NODE_ENV === 'production' ? 'require' : false,
});

// Create Drizzle database instance with schema
export const db: PostgresJsDatabase<typeof schema> = drizzle(client, {
  schema,
  logger: import.meta.env.NODE_ENV === 'development'
});

// Enhanced connection testing function
export async function testConnection(): Promise<boolean> {
  try {
    // Test basic connection
    const result = await client`SELECT version(), now() as current_time`;
    console.log('✅ PostgreSQL Connection Successful:', result[0]);
    
    // Test pgvector extension
    const vectorTest = await client`SELECT * FROM pg_extension WHERE extname = 'vector'`;
    if (vectorTest.length > 0) {
      console.log('✅ pgvector Extension Available:', vectorTest[0]);
    } else {
      console.warn('⚠️ pgvector Extension Not Found - Vector operations may fail');
    }
    
    // Test table existence
    const tables = await client`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY table_name
    `;
    console.log('✅ Available Tables:', tables.map(t => t.table_name));
    
    return true;
  } catch (error) {
    console.error('❌ Database Connection Failed:', error);
    return false;
  }
}

// Enhanced health check with vector capabilities
export async function getDatabaseHealth(): Promise<{
  connected: boolean;
  pgvectorEnabled: boolean;
  tablesCount: number;
  version: string;
  uptime?: string;
}> {
  try {
    // Basic connection test
    const versionResult = await client`SELECT version()`;
    const version = versionResult[0]?.version || 'Unknown';
    
    // Check pgvector extension
    const vectorCheck = await client`
      SELECT 1 FROM pg_extension WHERE extname = 'vector'
    `;
    const pgvectorEnabled = vectorCheck.length > 0;
    
    // Count tables
    const tableCount = await client`
      SELECT COUNT(*) as count 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
    `;
    const tablesCount = parseInt(tableCount[0]?.count || '0');
    
    // Database uptime
    const uptimeResult = await client`
      SELECT date_trunc('second', current_timestamp - pg_postmaster_start_time()) as uptime
    `;
    const uptime = uptimeResult[0]?.uptime;
    
    return {
      connected: true,
      pgvectorEnabled,
      tablesCount,
      version,
      uptime: uptime?.toString()
    };
    
  } catch (error) {
    console.error('Database health check failed:', error);
    return {
      connected: false,
      pgvectorEnabled: false,
      tablesCount: 0,
      version: 'Unknown'
    };
  }
}

// Vector operations helper functions
export class VectorOperations {
  /**
   * Calculate cosine similarity between two vectors
   */
  static async cosineSimilarity(vector1: number[], vector2: number[]): Promise<number> {
    try {
      const result = await client`
        SELECT (${vector1})::vector <=> (${vector2})::vector as similarity
      `;
      return 1 - parseFloat(result[0]?.similarity || '1'); // Convert distance to similarity
    } catch (error) {
      console.error('Cosine similarity calculation failed:', error);
      return 0;
    }
  }
  
  /**
   * Find similar vectors using cosine distance
   */
  static async findSimilarVectors(
    tableName: string,
    vectorColumn: string,
    queryVector: number[],
    limit: number = 10,
    threshold: number = 0.7
  ): Promise<any[]> {
    try {
      const result = await client`
        SELECT *, 
               1 - (${vectorColumn} <=> ${queryVector}::vector) as similarity_score
        FROM ${client(tableName)}
        WHERE 1 - (${vectorColumn} <=> ${queryVector}::vector) > ${threshold}
        ORDER BY ${vectorColumn} <=> ${queryVector}::vector
        LIMIT ${limit}
      `;
      return result;
    } catch (error) {
      console.error('Vector similarity search failed:', error);
      return [];
    }
  }
}

// Connection cleanup
export async function closeConnection(): Promise<void> {
  try {
    await client.end();
    console.log('✅ Database connection closed');
  } catch (error) {
    console.error('❌ Error closing database connection:', error);
  }
}

// Default export for convenience
export default db;

// Type exports for external use
export type { PostgresJsDatabase } from 'drizzle-orm/postgres-js';
export type Database = typeof db;

// Re-export schema for convenience
export * from './schema.js';
