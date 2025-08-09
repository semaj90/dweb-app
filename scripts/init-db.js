import pg from 'pg';
const client = new pg.Client({
  host: 'localhost',
  database: 'legal_ai',
  user: 'postgres',
  password: 'postgres'
});
await client.connect();
await client.query(`
  CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding JSONB
  )
`);
await client.end();
console.log('DB initialized');