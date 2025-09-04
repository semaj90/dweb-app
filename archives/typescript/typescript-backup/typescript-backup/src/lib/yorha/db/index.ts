// Database Connection Configuration
import { drizzle } from 'drizzle-orm/node-postgres';
import postgres from 'postgres';
import { migrate } from 'drizzle-orm/node-postgres/migrator';
import * as schema from './schema';
import { env } from '$env/dynamic/private';

// PostgreSQL connection
const connectionString = env.DATABASE_URL || 'postgresql://yorha:yorha_password@localhost:5432/yorha_db';

// For query purposes
const queryClient = postgres(connectionString);
export const db = drizzle(queryClient, { schema });

// For migrations
const migrationClient = postgres(connectionString, { max: 1 });
const migrationDb = drizzle(migrationClient, { schema });

export async function runMigrations(): Promise<any> {
  console.log('⏳ Running migrations...');
  
  try {
    await migrate(migrationDb, { migrationsFolder: './drizzle' });
    console.log('✅ Migrations completed successfully');
  } catch (error: any) {
    console.error('❌ Migration failed:', error);
    throw error;
  } finally {
    await migrationClient.end();
  }
}

// pgvector extension setup
export async function setupPgVector(): Promise<any> {
  try {
    await queryClient`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log('✅ pgvector extension enabled');
  } catch (error: any) {
    console.error('❌ Failed to enable pgvector:', error);
    throw error;
  }
}

// Initialize database
export async function initializeDatabase(): Promise<any> {
  await setupPgVector();
  await runMigrations();
}