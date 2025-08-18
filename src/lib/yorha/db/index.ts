// Database Connection Configuration
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
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

export async function runMigrations() {
  console.log('⏳ Running migrations...');
  
  try {
    await migrate(migrationDb, { migrationsFolder: './drizzle' });
    console.log('✅ Migrations completed successfully');
  } catch (error) {
    console.error('❌ Migration failed:', error);
    throw error;
  } finally {
    await migrationClient.end();
  }
}

// pgvector extension setup
export async function setupPgVector() {
  try {
    await queryClient`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log('✅ pgvector extension enabled');
  } catch (error) {
    console.error('❌ Failed to enable pgvector:', error);
    throw error;
  }
}

// Initialize database
export async function initializeDatabase() {
  await setupPgVector();
  await runMigrations();
}