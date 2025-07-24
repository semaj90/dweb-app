import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';

// PostgreSQL connection configuration
const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/deeds_db';

let db: any;
let client: any;

try {
  // Create PostgreSQL client
  client = postgres(connectionString, {
    max: 10,
    idle_timeout: 20,
    connect_timeout: 10,
  });
  
  // Initialize Drizzle with PostgreSQL
  db = drizzle(client, { schema });
  
  console.log('✅ PostgreSQL database connected');
} catch (error) {
  console.error('❌ PostgreSQL connection error:', error);
  
  // Fallback connection attempt
  try {
    client = postgres(connectionString, {
      max: 1,
      idle_timeout: 10,
      connect_timeout: 5,
    });
    db = drizzle(client, { schema });
    console.log('⚠️ Using fallback PostgreSQL connection');
  } catch (fallbackError) {
    console.error('❌ Fallback connection failed:', fallbackError);
    throw new Error('Unable to connect to PostgreSQL database');
  }
}

// Helper function to check if we're using PostgreSQL
export const isPostgreSQL = true;

// Export the database instance
export { db };

// Export the client for direct queries if needed
export { client };

// Graceful shutdown
process.on('beforeExit', () => {
  if (client) {
    client.end();
  }
});
