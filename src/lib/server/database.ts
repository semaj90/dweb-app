// Database connection and utilities for server-side operations
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { env } from '$env/dynamic/private';
import * as schema from './db/schema-postgres';

// Database configuration
const connectionString = env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db';

// Create the connection
const connection = postgres(connectionString, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10
});

// Create Drizzle instance
export const db = drizzle(connection, { schema });

// Export schema for convenience
export * from './db/schema-postgres';

// Connection utilities
export async function testConnection(): Promise<any> {
  try {
    await connection`SELECT 1`;
    return true;
  } catch (error: any) {
    console.error('Database connection test failed:', error);
    return false;
  }
}

export async function closeConnection(): Promise<any> {
  await connection.end();
}