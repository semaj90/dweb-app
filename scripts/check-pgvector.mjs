#!/usr/bin/env node
// Check PostgreSQL + pgvector readiness: todos table, legal_documents embedding dims, IVFFLAT index.
import { Client } from 'pg';

const DSN = process.env.DATABASE_URL || 'postgresql://postgres:password@127.0.0.1:5432/postgres';

(async () => {
  const pg = new Client({ connectionString: DSN });
  try {
    await pg.connect();
    console.log('✅ Connected to PostgreSQL');
    const tables = await pg.query("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1 LIMIT 50");
    console.log('📋 Tables:', tables.rows.map(r=>r.table_name).join(', '));

    // todos structure
    try {
      const todoCols = await pg.query("SELECT column_name,data_type FROM information_schema.columns WHERE table_name='todos'");
      console.log('\n📝 todos columns:');
      for (const c of todoCols.rows) console.log(' -', c.column_name, c.data_type);
    } catch { console.log('⚠️ todos table not found'); }

    // legal_documents embedding dims (pgvector)
    try {
      const dimRes = await pg.query("SELECT vector_dims(embedding) AS dims FROM legal_documents LIMIT 1");
      if (dimRes.rows.length) console.log(`\n🧬 legal_documents.embedding dims: ${dimRes.rows[0].dims}`);
      else console.log('\nℹ️ legal_documents empty or no embedding column');
    } catch (e){ console.log('ℹ️ Could not query embedding dims (maybe table/pgvector missing):', e.message); }

    // Index check
    try {
      const idx = await pg.query("SELECT indexname,indexdef FROM pg_indexes WHERE tablename='legal_documents' AND indexdef ILIKE '%ivfflat%' LIMIT 5");
      if (idx.rows.length){
        console.log('\n🔎 IVFFLAT indexes:');
        idx.rows.forEach(r=> console.log(' -', r.indexname));
      } else {
        console.log('\n⚠️ No IVFFLAT index detected on legal_documents');
      }
    } catch(e){ console.log('ℹ️ Index check failed:', e.message); }
  } catch (e) {
    console.error('❌ PostgreSQL connection failed:', e.message);
  } finally {
    await pg.end();
  }
})();
