import { drizzle } from 'drizzle-orm/better-sqlite3';
import Database from 'better-sqlite3';
import * as schema from './schema';

let db: any;

try {
  const dbPath = process.env.DATABASE_URL || './dev.db';
  const sqlite = new Database(dbPath);
  db = drizzle(sqlite, { schema });
  console.log('✅ Database connected');
} catch (error) {
  console.error('❌ Database error:', error);
  const sqlite = new Database(':memory:');
  db = drizzle(sqlite, { schema });
  console.log('⚠️ Using in-memory database');
}

export { db };
