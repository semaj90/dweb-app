const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.PG_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db' });
  await c.connect();
  const count = await c.query('SELECT COUNT(*)::int AS n FROM legal_documents');
  console.log('legal_documents count =', count.rows[0].n);
  const r = await c.query(`SELECT id, title, LEFT(COALESCE(full_text, content)::text, 200) AS snippet FROM legal_documents WHERE COALESCE(title,'') <> '' ORDER BY created_at DESC NULLS LAST LIMIT 10`);
  r.rows.forEach((row, i) => {
    console.log(`${i+1}. ${row.title}`);
    console.log('   snippet:', (row.snippet||'').replace(/\s+/g,' ').slice(0, 200));
  });
  await c.end();
})().catch(e => { console.error('ERR', e); process.exit(1); });
