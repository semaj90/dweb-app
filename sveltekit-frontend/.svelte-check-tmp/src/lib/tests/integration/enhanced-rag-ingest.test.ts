import { describe, it, expect } from 'vitest';

const base = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

describe('Enhanced RAG Ingestion API', () => {
  it('POST /api/enhanced-rag/ingest queues ids', async () => {
    const ids = ['smoke-test-a', 'smoke-test-b'];
    const res = await fetch(`${base}/api/enhanced-rag/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids, mode: 'incremental' })
    });
    expect(res.ok).toBe(true);
    const json = await res.json();
    expect(json.success).toBe(true);
    expect(json.queued).toBeGreaterThanOrEqual(ids.length);
  });
});
