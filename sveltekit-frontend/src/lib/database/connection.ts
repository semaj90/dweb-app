
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from 'pg';
import * as schema from './schema';

const env = { 
  DATABASE_URL: process.env.DATABASE_URL || import.meta.env?.DATABASE_URL 
};

// Create PostgreSQL connection pool
const pool = new Pool({
  connectionString: env.DATABASE_URL || 'postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_db',
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Create Drizzle instance with schema
export const db = drizzle(pool, { schema });

// Connection health check
export async function testDatabaseConnection(): Promise<{
  success: boolean;
  message: string;
  details?: unknown;
}> {
  try {
    // Test basic connection
    const client = await pool.connect();
    const result = await client.query('SELECT version();');
    client.release();

    // Test pgvector extension
    const vectorTest = await pool.query("SELECT extname FROM pg_extension WHERE extname = 'vector';");
    const hasVector = vectorTest.rows.length > 0;

    return {
      success: true,
      message: 'Database connection successful',
      details: {
        postgresVersion: result.rows[0]?.version,
        pgvectorEnabled: hasVector,
        poolSize: pool.totalCount,
        timestamp: new Date().toISOString(),
      },
    };
  } catch (error) {
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
) {
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

    const result = await pool.query(query, [
      JSON.stringify(queryEmbedding),
      threshold,
      limit,
    ]);

    return {
      success: true,
      results: result.rows,
      count: result.rows.length,
    };
  } catch (error) {
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
) {
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

    const result = await pool.query(searchQuery, params);

    return {
      success: true,
      results: result.rows,
      count: result.rows.length,
      query,
      queryEmbedding: queryEmbedding.slice(0, 5), // Only return first 5 dimensions for debugging
    };
  } catch (error) {
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
export async function initializeDatabase() {
  try {
    console.log('🔄 Initializing database...');

    // Create extensions
    await pool.query('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";');
    await pool.query('CREATE EXTENSION IF NOT EXISTS vector;');

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
  } catch (error) {
    console.error('❌ Database initialization error:', error);
    return {
      success: false,
      message: `Initialization failed: ${(error as Error).message}`,
    };
  }
}

// Graceful shutdown
export async function closeDatabaseConnection() {
  try {
    await pool.end();
    console.log('✅ Database connection pool closed');
  } catch (error) {
    console.error('❌ Error closing database connection:', error);
  }
}

// Export the pool for direct access if needed
export { pool };