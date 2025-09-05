import fs from 'fs';
import fetch from 'node-fetch';

const ROUTES_FILE = './scripts/routes-to-test.json';
const OUTPUT_FILE = './scripts/route-smoke-results.jsonl';

async function run() {
  if (!fs.existsSync(ROUTES_FILE)) {
    console.error('Missing routes file:', ROUTES_FILE);
    process.exit(2);
  }

  const routes = JSON.parse(fs.readFileSync(ROUTES_FILE, 'utf8'));
  const out = fs.createWriteStream(OUTPUT_FILE, { flags: 'w' });

  for (const r of routes) {
    const url = r.url.replace('<HOST>', process.env.HOST || 'http://localhost:5173');
    const method = (r.method || 'GET').toUpperCase();
    const t0 = Date.now();
    try {
      const res = await fetch(url, { method, redirect: 'manual' });
      const text = await res.text().catch(() => '');
      const snippet = text.slice(0, 1000);
      const headers = {};
      res.headers.forEach((v, k) => headers[k] = v);

      const rec = {
        ts: new Date().toISOString(),
        route: r.url,
        method,
        url,
        status: res.status,
        redirected: res.status >= 300 && res.status < 400,
        headers,
        bodySnippet: snippet,
        durationMs: Date.now() - t0
      };

      out.write(JSON.stringify(rec) + '\n');
      console.log(`${method} ${url} -> ${res.status} ${rec.durationMs}ms`);
    } catch (err) {
      const rec = {
        ts: new Date().toISOString(),
        route: r.url,
        method,
        url,
        error: String(err),
        durationMs: Date.now() - t0
      };
      out.write(JSON.stringify(rec) + '\n');
      console.error(`${method} ${url} -> ERROR:`, err.message || err);
    }
  }

  out.end();
  console.log('Results written to', OUTPUT_FILE);
}

run().catch(err => { console.error(err); process.exit(1); });
