#!/usr/bin/env node
import pkg from 'pg';
const { Client } = pkg;

const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

(async () => {
  console.log('▶️ Running standalone DB connection test...');
  const client = new Client({ connectionString });
  try {
    await client.connect();
    const res = await client.query('SELECT 1 as ok');
    console.log('✅ Connected to DB, test query result:', res.rows[0]);
    await client.end();
    process.exit(0);
  } catch (err) {
    console.error('❌ DB connection failed:', err.message || err);
    try { await client.end(); } catch(e){}
    process.exit(2);
  }
})();
