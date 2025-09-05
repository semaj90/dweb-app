import { Client } from 'pg';

const dsn = process.env.DATABASE_URL || 'postgresql://postgres:password@127.0.0.1:5432/postgres';
const table = process.env.RAG_TABLE || 'legal_documents';
const col = process.env.RAG_VECTOR_COL || 'embedding';
const lists = process.env.PGVECTOR_LISTS || 100;

async function run(){
  const client = new Client({ connectionString: dsn });
  await client.connect();
  await client.query('CREATE EXTENSION IF NOT EXISTS vector');
  await client.query(`DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='${table}' AND column_name='${col}') THEN EXECUTE 'ALTER TABLE ${table} ADD COLUMN ${col} vector(768)'; END IF; END $$;`);
  await client.query(`CREATE INDEX IF NOT EXISTS ${table}_${col}_ivfflat ON ${table} USING ivfflat (${col} vector_cosine_ops) WITH (lists = ${lists});`);
  console.log('✅ pgvector IVFFLAT ensured');
  await client.end();
}
run().catch(e=>{ console.error('Index creation failed', e); process.exit(1); });
