// Enhanced PostgreSQL Database Service with Drizzle ORM
// Provides type-safe database operations with proper TypeScript support

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import * as schema from './schema/legal-documents.js';

// Database configuration
const connectionString = process.env.DATABASE_URL || 
  `postgresql://${process.env.DB_USER || 'postgres'}:${process.env.DB_PASSWORD || 'postgres'}@${process.env.DB_HOST || 'localhost'}:${process.env.DB_PORT || 5432}/${process.env.DB_NAME || 'legal_ai'}`;

// Create postgres client
const queryClient = postgres(connectionString, {
  max: 20,
  idle_timeout: 30,
  connect_timeout: 10,
  ssl: process.env.NODE_ENV === 'production' ? 'require' : false
});

// Create Drizzle database instance
export const db = drizzle(queryClient, { schema });

// Export schema for use in API routes
export { schema };

// Re-export specific tables for easier imports
export const { 
  legalDocuments, 
  contentEmbeddings, 
  searchSessions, 
  embeddings 
} = schema;

// Database connection management
class DatabaseManager {
  private connected = false;
  
  async connect(): Promise<boolean> {
    try {
      // Test connection
      await queryClient`SELECT 1`;
      this.connected = true;
      console.log('Connected to PostgreSQL via Drizzle ORM');
      return true;
    } catch (error) {
      console.error('Failed to connect to PostgreSQL:', error);
      this.connected = false;
      return false;
    }
  }

  async migrate(): Promise<void> {
    try {
      // Run migrations
      await migrate(db, { migrationsFolder: './drizzle' });
      console.log('Database migrations completed');
    } catch (error) {
      console.error('Migration failed:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      await queryClient.end();
      this.connected = false;
      console.log('Disconnected from PostgreSQL');
    } catch (error) {
      console.error('Error disconnecting from database:', error);
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  async healthCheck(): Promise<{
    connected: boolean;
    responseTime?: number;
    error?: string;
  }> {
    try {
      const start = Date.now();
      await queryClient`SELECT 1`;
      const responseTime = Date.now() - start;
      
      return {
        connected: true,
        responseTime
      };
    } catch (error: any) {
      return {
        connected: false,
        error: error.message
      };
    }
  }
}

// Export database manager instance
export const dbManager = new DatabaseManager();

// Utility functions for common operations
export const dbUtils = {
  /**
   * Initialize database connection
   */
  async initialize(): Promise<boolean> {
    const connected = await dbManager.connect();
    if (connected) {
      try {
        // Run migrations if needed
        await dbManager.migrate();
        return true;
      } catch (error) {
        console.error('Failed to run migrations:', error);
        return false;
      }
    }
    return false;
  },

  /**
   * Execute raw SQL query
   */
  async executeRaw<T = any>(query: string, params: any[] = []): Promise<T[]> {
    try {
      const result = await queryClient.unsafe(query, params);
      // Safe type conversion - convert result to array if it isn't already
      const resultArray = Array.isArray(result) ? result : [result];
      return resultArray as T[];
    } catch (error) {
      console.error('Raw query execution failed:', error);
      throw error;
    }
  },

  /**
   * Check if table exists
   */
  async tableExists(tableName: string): Promise<boolean> {
    try {
      const result = await queryClient`
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = ${tableName}
        )
      `;
      return result[0]?.exists || false;
    } catch (error) {
      console.error(`Error checking if table ${tableName} exists:`, error);
      return false;
    }
  },

  /**
   * Get table row count
   */
  async getTableCount(tableName: string): Promise<number> {
    try {
      const result = await queryClient.unsafe(`SELECT COUNT(*) FROM ${tableName}`);
      return parseInt(result[0]?.count || '0');
    } catch (error) {
      console.error(`Error getting count for table ${tableName}:`, error);
      return 0;
    }
  },

  /**
   * Truncate table
   */
  async truncateTable(tableName: string): Promise<void> {
    try {
      await queryClient.unsafe(`TRUNCATE TABLE ${tableName} RESTART IDENTITY CASCADE`);
      console.log(`Table ${tableName} truncated successfully`);
    } catch (error) {
      console.error(`Error truncating table ${tableName}:`, error);
      throw error;
    }
  },

  /**
   * Create database backup
   */
  async createBackup(): Promise<string> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupName = `backup_${timestamp}`;
    
    try {
      // This would typically use pg_dump in a real implementation
      console.log(`Creating backup: ${backupName}`);
      return backupName;
    } catch (error) {
      console.error('Backup creation failed:', error);
      throw error;
    }
  }
};

// Export query client for direct access if needed
export { queryClient };

// Default export for backward compatibility
export default {
  db,
  dbManager,
  dbUtils,
  schema,
  queryClient
};

// Graceful shutdown handling
process.on('SIGINT', async () => {
  console.log('Shutting down database connection...');
  await dbManager.disconnect();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('Shutting down database connection...');
  await dbManager.disconnect();
  process.exit(0);
});
