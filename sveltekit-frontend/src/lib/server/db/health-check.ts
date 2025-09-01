/**
 * PostgreSQL + pgvector Health Check System
 * Validates database connectivity and vector extension availability
 */

import { db } from './index.js';
import { sql } from 'drizzle-orm';

export interface DatabaseHealthStatus {
  connected: boolean;
  pgvectorEnabled: boolean;
  version?: string;
  pgvectorVersion?: string;
  tablesCount?: number;
  error?: string;
  responseTime?: number;
  details?: {
    host?: string;
    port?: number;
    database?: string;
    ssl?: boolean;
  };
}

export class DatabaseHealthChecker {
  private static instance: DatabaseHealthChecker;
  private lastHealthCheck?: DatabaseHealthStatus;
  private lastCheckTime?: number;
  private readonly CACHE_DURATION = 30000; // 30 seconds

  static getInstance(): DatabaseHealthChecker {
    if (!DatabaseHealthChecker.instance) {
      DatabaseHealthChecker.instance = new DatabaseHealthChecker();
    }
    return DatabaseHealthChecker.instance;
  }

  /**
   * Comprehensive database health check
   */
  async checkHealth(useCache = true): Promise<DatabaseHealthStatus> {
    const now = Date.now();
    
    // Return cached result if recent and caching enabled
    if (useCache && this.lastHealthCheck && this.lastCheckTime && 
        (now - this.lastCheckTime) < this.CACHE_DURATION) {
      return this.lastHealthCheck;
    }

    const startTime = Date.now();
    let status: DatabaseHealthStatus = {
      connected: false,
      pgvectorEnabled: false
    };

    try {
      // Test basic connectivity
      const connectionTest = await db.execute(sql`SELECT 1 as test`);
      if (!connectionTest || connectionTest.length === 0) {
        throw new Error('Connection test failed');
      }

      status.connected = true;

      // Get PostgreSQL version
      const versionResult = await db.execute(sql`SELECT version() as version`);
      if (versionResult && versionResult[0]) {
        status.version = (versionResult[0] as any).version;
      }

      // Check for pgvector extension
      try {
        const pgvectorCheck = await db.execute(
          sql`SELECT extversion FROM pg_extension WHERE extname = 'vector'`
        );
        
        if (pgvectorCheck && pgvectorCheck.length > 0) {
          status.pgvectorEnabled = true;
          status.pgvectorVersion = (pgvectorCheck[0] as any).extversion;
        } else {
          // Try to create the extension if it doesn't exist (for dev environments)
          try {
            await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector`);
            
            // Check again after creation attempt
            const retryCheck = await db.execute(
              sql`SELECT extversion FROM pg_extension WHERE extname = 'vector'`
            );
            
            if (retryCheck && retryCheck.length > 0) {
              status.pgvectorEnabled = true;
              status.pgvectorVersion = (retryCheck[0] as any).extversion;
            }
          } catch (extensionError) {
            console.warn('Could not create pgvector extension:', extensionError);
            status.pgvectorEnabled = false;
          }
        }
      } catch (vectorError) {
        console.warn('pgvector check failed:', vectorError);
        status.pgvectorEnabled = false;
      }

      // Count tables to verify schema is loaded
      try {
        const tablesResult = await db.execute(
          sql`
            SELECT COUNT(*) as count 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
          `
        );
        
        if (tablesResult && tablesResult[0]) {
          status.tablesCount = parseInt((tablesResult[0] as any).count);
        }
      } catch (tablesError) {
        console.warn('Tables count check failed:', tablesError);
      }

      status.responseTime = Date.now() - startTime;

      // Add connection details
      status.details = {
        host: import.meta.env.DATABASE_HOST || 'localhost',
        port: parseInt(import.meta.env.DATABASE_PORT || '5432'),
        database: import.meta.env.DATABASE_NAME || 'legal_ai_db',
        ssl: Boolean(import.meta.env.DATABASE_SSL)
      };

    } catch (error: any) {
      status.connected = false;
      status.error = error instanceof Error ? error.message : String(error);
      status.responseTime = Date.now() - startTime;
    }

    // Cache the result
    this.lastHealthCheck = status;
    this.lastCheckTime = now;

    return status;
  }

  /**
   * Quick connectivity test (cached)
   */
  async isConnected(): Promise<boolean> {
    const health = await this.checkHealth(true);
    return health.connected;
  }

  /**
   * Check if pgvector is available
   */
  async isPgVectorEnabled(): Promise<boolean> {
    const health = await this.checkHealth(true);
    return health.pgvectorEnabled;
  }

  /**
   * Test vector operations
   */
  async testVectorOperations(): Promise<boolean> {
    if (!await this.isPgVectorEnabled()) {
      return false;
    }

    try {
      // Test basic vector operations
      const testVector = '[0.1, 0.2, 0.3]';
      const result = await db.execute(
        sql`SELECT ${testVector}::vector as test_vector`
      );
      
      if (!result || result.length === 0) {
        return false;
      }

      // Test cosine similarity
      const similarityTest = await db.execute(
        sql`SELECT ${testVector}::vector <=> ${testVector}::vector as similarity`
      );

      return similarityTest && similarityTest.length > 0;
    } catch (error: any) {
      console.warn('Vector operations test failed:', error);
      return false;
    }
  }

  /**
   * Validate essential tables exist
   */
  async validateSchema(): Promise<{ valid: boolean; missingTables: string[] }> {
    const essentialTables = [
      'users', 'sessions', 'cases', 'evidence', 'criminals', 
      'legal_documents', 'persons_of_interest', 'content_embeddings'
    ];

    const missingTables: string[] = [];

    try {
      for (const tableName of essentialTables) {
        const tableCheck = await db.execute(
          sql`
            SELECT EXISTS (
              SELECT FROM information_schema.tables 
              WHERE table_schema = 'public' 
              AND table_name = ${tableName}
            ) as exists
          `
        );

        if (!tableCheck || !tableCheck[0] || !(tableCheck[0] as any).exists) {
          missingTables.push(tableName);
        }
      }
    } catch (error: any) {
      console.error('Schema validation failed:', error);
      return { valid: false, missingTables: essentialTables };
    }

    return { valid: missingTables.length === 0, missingTables };
  }

  /**
   * Get detailed database metrics
   */
  async getDatabaseMetrics(): Promise<any> {
    try {
      const health = await this.checkHealth(false); // Force fresh check
      
      if (!health.connected) {
        return { error: 'Database not connected', health };
      }

      // Get database size
      const sizeResult = await db.execute(
        sql`SELECT pg_size_pretty(pg_database_size(current_database())) as size`
      );

      // Get connection count
      const connectionsResult = await db.execute(
        sql`
          SELECT count(*) as total_connections,
                 count(*) FILTER (WHERE state = 'active') as active_connections
          FROM pg_stat_activity
          WHERE datname = current_database()
        `
      );

      // Get top tables by size
      const tablesResult = await db.execute(
        sql`
          SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
          FROM pg_tables
          WHERE schemaname = 'public'
          ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
          LIMIT 10
        `
      );

      return {
        health,
        database: {
          size: sizeResult?.[0] ? (sizeResult[0] as any).size : 'Unknown',
          connections: connectionsResult?.[0] || { total_connections: 0, active_connections: 0 },
          topTables: tablesResult || []
        },
        timestamp: new Date().toISOString()
      };

    } catch (error: any) {
      return { 
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Clear health check cache
   */
  clearCache(): void {
    this.lastHealthCheck = undefined;
    this.lastCheckTime = undefined;
  }
}

// Export singleton instance
export const dbHealthChecker = DatabaseHealthChecker.getInstance();
;
/**
 * Simple health check function for quick use
 */
export async function checkDatabaseHealth(): Promise<DatabaseHealthStatus> {
  return dbHealthChecker.checkHealth();
}

/**
 * Startup validation function
 */
export async function validateDatabaseOnStartup(): Promise<boolean> {
  console.log('🔍 Checking database connectivity...');
  
  const health = await dbHealthChecker.checkHealth(false);
  
  if (health.connected) {
    console.log('✅ PostgreSQL connected successfully');
    console.log(`📊 Version: ${health.version}`);
    console.log(`📈 Response time: ${health.responseTime}ms`);
    console.log(`🗃️ Tables found: ${health.tablesCount || 'Unknown'}`);
    
    if (health.pgvectorEnabled) {
      console.log(`🧮 pgvector enabled: v${health.pgvectorVersion}`);
      
      // Test vector operations
      const vectorTest = await dbHealthChecker.testVectorOperations();
      if (vectorTest) {
        console.log('✅ Vector operations working');
      } else {
        console.log('⚠️ Vector operations test failed');
      }
    } else {
      console.log('⚠️ pgvector extension not found');
    }

    // Validate schema
    const schema = await dbHealthChecker.validateSchema();
    if (schema.valid) {
      console.log('✅ Database schema validated');
    } else {
      console.log('⚠️ Missing tables:', schema.missingTables.join(', '));
    }

    return true;
  } else {
    console.log('❌ PostgreSQL connection failed:', health.error);
    console.log('📋 Connection details:', health.details);
    return false;
  }
}