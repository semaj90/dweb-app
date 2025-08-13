const { Client } = require('pg');
(async () => {
  const c = new Client({ connectionString: process.env.PG_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db' });
  await c.connect();
  const cols = await c.query("SELECT column_name FROM information_schema.columns WHERE table_name='legal_documents' ORDER BY ordinal_position");
  console.log('Columns:', cols.rows.map(r => r.column_name).join(', '));
  const sample = await c.query('SELECT * FROM legal_documents LIMIT 1');
  console.log('Sample row keys:', Object.keys(sample.rows[0]||{}));
  await c.end();
})().catch(e => { console.error('ERR', e); process.exit(1); });
