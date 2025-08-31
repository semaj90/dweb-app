// Minimal runtime API testing suite (placeholder)
// Purpose: perform a simple create/read/update/delete against local API endpoints.
// Replace endpoints and payloads with your real API routes.

async function request(path, opts = {}) {
  const url = `http://localhost:5173${path}`;
  const res = await fetch(url, opts);
  const text = await res.text();
  try { return { ok: res.ok, status: res.status, body: JSON.parse(text) }; } catch { return { ok: res.ok, status: res.status, body: text }; }
}

(async () => {
  console.log('▶️ Running CRUD smoke tests against http://localhost:5173');
  // Create
  const create = await request('/api/test-items', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'test-item', content: 'automated test' }) });
  console.log('CREATE:', create.status, create.ok);

  const id = create.body?.id || create.body?.insertedId || 'sample-id';

  // Read
  const read = await request(`/api/test-items/${id}`);
  console.log('READ:', read.status, read.ok);

  // Update
  const update = await request(`/api/test-items/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'updated' }) });
  console.log('UPDATE:', update.status, update.ok);

  // Delete
  const del = await request(`/api/test-items/${id}`, { method: 'DELETE' });
  console.log('DELETE:', del.status, del.ok);

  const passed = create.ok && read.ok && update.ok && del.ok;
  console.log(passed ? '✅ CRUD smoke tests passed' : '❌ CRUD smoke tests failed');
  process.exit(passed ? 0 : 1);
})();
