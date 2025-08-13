const { Client } = require('pg');
const http = require('node:http');

async function postJSON(url, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const data = Buffer.from(JSON.stringify(body));
    const req = http.request({
      hostname: u.hostname,
      port: u.port || 80,
      path: u.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': data.length },
    }, (res) => {
      let out = '';
      res.setEncoding('utf8');
      res.on('data', (d) => out += d);
      res.on('end', () => {
        try { resolve(JSON.parse(out)); } catch(e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

(async () => {
  const prompt = process.argv[2] || 'contract liability terms';
  const emb = await postJSON('http://localhost:11434/api/embeddings', { model: 'nomic-embed-text', input: prompt });
  const vec = emb.embedding || (emb.data && emb.data[0] && emb.data[0].embedding);
  if (!Array.isArray(vec)) throw new Error('No embedding vector');
  console.log('Embedding dims:', vec.length);
  const veclit = '[' + vec.join(',') + ']';
  const c = new Client({ connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db' });
  await c.connect();
  const q = `SELECT id, title, 1 - (embedding <=> '${veclit}'::vector) AS similarity FROM legal_documents WHERE embedding IS NOT NULL ORDER BY embedding <=> '${veclit}'::vector LIMIT 5`;
  const r = await c.query(q);
  console.log(r.rows);
  await c.end();
})().catch(e => { console.error('ERR', e); process.exit(1); });
