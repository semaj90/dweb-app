import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './src/lib/server/db/schema-postgres.ts';

const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db';

console.log('🔗 Connecting to database...');
const connection = postgres(connectionString);
const db = drizzle(connection, { schema });

console.log('✅ Connected to database successfully!');
console.log('📊 Database setup complete. Schema is ready for use.');

// Test basic query
try {
  const result = await connection`SELECT version()`;
  console.log('🎉 Database version:', result[0].version.slice(0, 50) + '...');
} catch (error) {
  console.log('❌ Error testing connection:', error.message);
}

await connection.end();
console.log('👋 Connection closed.');