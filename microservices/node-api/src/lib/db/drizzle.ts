import { Pool } from 'pg';
import { drizzle } from 'drizzle-orm/node-postgres';

const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/legal_ai_db';

const pool = new Pool({ connectionString, max: 5 });

export const db = drizzle(pool);

export async function healthCheck() {
  const client = await pool.connect();
  try {
    await client.query('select 1');
    return true;
  } finally {
    client.release();
  }
}
