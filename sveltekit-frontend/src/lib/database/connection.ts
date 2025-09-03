/// <reference types="vite/client" />

import { drizzle } from "drizzle-orm/postgres-js";
import postgres from 'postgres';
import * as schema from './schema.js';

// Get DATABASE_URL from environment with fallback
const DATABASE_URL = import.meta.env.VITE_DATABASE_URL ||
  import.meta.env.DATABASE_URL ||
  'postgresql://postgres:123456@localhost:5432/legal_ai_db';

// Create PostgreSQL connection using postgres.js
const sql = postgres(DATABASE_URL, {
  max: 10,
  idle_timeout: 30,
  connect_timeout: 2,
});

// Create Drizzle instance with schema
export const db = drizzle(sql, { schema });

// Export sql connection for direct queries
export const pool = sql; // alias for consistency (postgres.js instance)
// Connection health check
export async function testDatabaseConnection(): Promise<{
  success: boolean;
  message: string;
  details?: unknown;
}> {
  try {
    // Test basic connection
    const result = await pool`SELECT version();`;

    // Test pgvector extension
    const vectorTest = await pool`SELECT extname FROM pg_extension WHERE extname = 'vector';`;
    const hasVector = Array.isArray(vectorTest) && vectorTest.length > 0;

    return {
      success: true,
      message: 'Database connection successful',
      details: {
        postgresVersion: (result as any)[0]?.version,
        pgvectorEnabled: hasVector,
        poolSize: 'n/a',
        timestamp: new Date().toISOString(),
      },
    };
  } catch (error: any) {
    return {
      success: false,
      message: `Database connection failed: ${(error as Error).message}`,
      details: {
        error: (error as Error).stack,
        timestamp: new Date().toISOString(),
      },
    };
  }
}

// Vector similarity search helper
export async function vectorSimilaritySearch(
  table: 'documents' | 'search_index',
  queryEmbedding: number[],
  limit: number = 10,
  threshold: number = 0.8
): Promise<any> {
  try {
    const tableName = table === 'documents' ? 'documents' : 'search_index';

    // Use pgvector's cosine distance operator
    const query = `
      SELECT *, 1 - (embedding <=> $1) AS similarity
      FROM ${tableName}
      WHERE 1 - (embedding <=> $1) > $2
      ORDER BY embedding <=> $1
      LIMIT $3
    `;

    const result = await pool.unsafe(query, [
      JSON.stringify(queryEmbedding),
      threshold,
      limit,
    ]);

    return {
      success: true,
      results: result,
      count: Array.isArray(result) ? result.length : 0,
    };
  } catch (error: any) {
    return {
      success: false,
      error: (error as Error).message,
      results: [],
      count: 0,
    };
  }
}

// Hybrid semantic search combining multiple tables
export async function hybridSemanticSearch(
  query: string,
  queryEmbedding: number[],
  options: {
    limit?: number;
    threshold?: number;
    entityTypes?: string[];
    caseId?: string;
    userId?: string;
  } = {}
): Promise<any> {
  const { limit = 10, threshold = 0.7, entityTypes, caseId, userId } = options;

  try {
    let whereClause = `WHERE 1 - (si.embedding <=> $1) > $2`;
    const params: any[] = [JSON.stringify(queryEmbedding), threshold];
    let paramIndex = 2;

    if (entityTypes && entityTypes.length > 0) {
      paramIndex++;
      whereClause += ` AND si.entity_type = ANY($${paramIndex})`;
      params.push(entityTypes);
    }

    if (caseId) {
      paramIndex++;
      whereClause += ` AND (
        (si.entity_type = 'case' AND si.entity_id = $${paramIndex}) OR
        (si.entity_type = 'document' AND EXISTS (
          SELECT 1 FROM documents d WHERE d.id = si.entity_id AND d.case_id = $${paramIndex}
        )) OR
        (si.entity_type = 'evidence' AND EXISTS (
          SELECT 1 FROM evidence e WHERE e.id = si.entity_id AND e.case_id = $${paramIndex}
        ))
      )`;
      params.push(caseId);
    }

    const searchQuery = `
      SELECT
        si.*,
        1 - (si.embedding <=> $1) AS similarity,
        CASE si.entity_type
          WHEN 'document' THEN d.title
          WHEN 'evidence' THEN e.title
          WHEN 'case' THEN c.title
          ELSE si.metadata->>'title'
        END AS entity_title
      FROM search_index si
      LEFT JOIN documents d ON si.entity_type = 'document' AND si.entity_id = d.id
      LEFT JOIN evidence e ON si.entity_type = 'evidence' AND si.entity_id = e.id
      LEFT JOIN cases c ON si.entity_type = 'case' AND si.entity_id = c.id
      ${whereClause}
      ORDER BY si.embedding <=> $1
      LIMIT $${paramIndex + 1}
    `;

    params.push(limit);

    const result = await pool.unsafe(searchQuery, params);

    return {
      success: true,
      results: result,
      count: Array.isArray(result) ? result.length : 0,
      query,
      queryEmbedding: queryEmbedding.slice(0, 5),
    };
  } catch (error: any) {
    return {
      success: false,
      error: (error as Error).message,
      results: [],
      count: 0,
      query,
    };
  }
}

// Initialize database with extensions and basic setup
export async function initializeDatabase(): Promise<any> {
  try {
    console.log('🔄 Initializing database...');

    // Create extensions
    await pool`CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`;
    await pool`CREATE EXTENSION IF NOT EXISTS vector;`;

    console.log('✅ Database extensions created');

    // Run migrations would go here
    // await migrate(db, { migrationsFolder: './drizzle' });

    const health = await testDatabaseConnection();
    if (health.success) {
      console.log('✅ Database initialization complete');
      console.log('📊 Database details:', health.details);
    } else {
      console.error('❌ Database initialization failed:', health.message);
    }

    return health;
  } catch (error: any) {
    console.error('❌ Database initialization error:', error);
    return {
      success: false,
      message: `Initialization failed: ${(error as Error).message}`,
    };
  }
}

// Graceful shutdown
export async function closeDatabaseConnection(): Promise<any> {
  try {
    await pool.end();
    console.log('✅ Database connection pool closed');
  } catch (error: any) {
    console.error('❌ Error closing database connection:', error);
  }
}

// Note: pool exported once at declaration to avoid duplicate export errors

// Direct SQL for complex vector operations
export async function executeSQL(query: string, params: any[] = []) {
  try {
    const result = await pool.unsafe(query, params);
    return { success: true, data: result, rowCount: Array.isArray(result) ? result.length : 0 };
  } catch (error: any) {
    return { success: false, error: error.message };
  }
}