import { Pool } from 'pg';

const connectionString = process.env.PG_CONNECTION_STRING || 'postgresql://postgres:postgres@localhost:5432/legal_ai_db';
const pool = new Pool({ connectionString });

export async function queryDb(text: string, params: any[] = []) {
  const client = await pool.connect();
  try {
    const res = await client.query(text, params);
    return res.rows;
  } finally {
    client.release();
  }
}
