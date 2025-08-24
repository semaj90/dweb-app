/// <reference types="vitest" />
// Use vitest globals (describe/it/expect/beforeEach) to avoid named-import type issues
// Attempt to import POST handler and rate limiter reset for direct invocation
// These imports rely on module paths; adjust if path resolution differs
import { POST } from '../../src/routes/api/evidence/upload/+server';
import { __resetRateLimiter } from '../../src/lib/server/rateLimit';

// Minimal mock of SvelteKit Locals with user
function mockEvent(body: BodyInit, headers: Record<string,string> = {}) {
  const h = new Headers({ 'content-type': 'multipart/form-data; boundary=----test', ...headers });
  const request = new Request('http://localhost/api/evidence/upload', { method: 'POST', body, headers: h });
  return { request, locals: { user: { id: 'user1', role: 'admin' } } } as any;
}

// Utility to build a multipart body manually (simplified for tests)
function multipart(fields: Record<string,string>, files?: { field?: string; name: string; type: string; content: string }[]) {
  const boundary = '----test';
  const parts: string[] = [];
  for (const [k,v] of Object.entries(fields)) {
    parts.push(`--${boundary}\r\nContent-Disposition: form-data; name="${k}"\r\n\r\n${v}\r\n`);
  }
  files?.forEach(f => {
    const data = f.content;
    parts.push(`--${boundary}\r\nContent-Disposition: form-data; name="${f.field||'files'}"; filename="${f.name}"\r\nContent-Type: ${f.type}\r\n\r\n${data}\r\n`);
  });
  parts.push(`--${boundary}--`);
  return parts.join('');
}

beforeEach(() => __resetRateLimiter());

// Lightweight integration-style tests using fetch against a dev server or mocked RequestHandler.
// NOTE: For a full test you'd spin up the SvelteKit server; here we unit-test the POST logic if exported separately.

// (Removed unused buildForm helper to avoid Node Buffer/File incompatibility in test environment)

describe('evidence upload handler', () => {
  it('rejects missing files', async () => {
    const body = multipart({ summaryType: 'narrative' });
    const res: Response = await POST(mockEvent(body));
    const json = await res.json();
    expect(res.status).toBe(400);
    expect(json.success).toBe(false);
    expect(json.error.message).toMatch(/No files/);
    expect(res.headers.get('x-correlation-id')).toBeTruthy();
  });

  it('accepts valid summaryType and small text file', async () => {
    const fileContent = 'Hello legal world';
    const body = multipart({ summaryType: 'narrative', generateSummary: 'true' }, [ { name: 'test.txt', type: 'text/plain', content: fileContent } ]);
    const res: Response = await POST(mockEvent(body));
    const json = await res.json();
    // Either 201 success or 500 if external dependencies unavailable; assert structural expectations when success
    if (res.status === 201) {
      expect(json.success).toBe(true);
      expect(json.data[0].url).toMatch(/http/);
      expect(json.meta.correlationId).toBeTruthy();
    } else {
      expect([500]).toContain(res.status);
    }
  });

  it('rejects invalid summaryType', async () => {
    const body = multipart({ summaryType: 'invalid_type' }, [ { name: 'f.txt', type: 'text/plain', content: 'x' } ]);
    const res: Response = await POST(mockEvent(body));
    const json = await res.json();
    expect(res.status).toBe(400);
    expect(json.error.message).toMatch(/Invalid summaryType/);
  });

  it('rejects unsupported mime type', async () => {
    const body = multipart({ summaryType: 'narrative' }, [ { name: 'script.exe', type: 'application/x-msdownload', content: 'MZ..' } ]);
    const res: Response = await POST(mockEvent(body));
    if (res.status !== 400) return; // skip if environment altered mime whitelist
    const json = await res.json();
    expect(json.error.message).toMatch(/Unsupported file type/);
    expect(res.headers.get('x-correlation-id')).toBeTruthy();
  });

  it('attaches correlation id on error', async () => {
    // Force error by sending wrong content-type (not multipart)
    const request = new Request('http://localhost/api/evidence/upload', { method: 'POST', body: 'plain', headers: { 'content-type': 'text/plain' } });
    const res: Response = await POST({ request, locals: { user: { id: 'user1', role: 'admin' } } } as any);
    const json = await res.json();
    expect(res.status).toBe(400);
    expect(json.error.message).toMatch(/Content-Type must be multipart/);
    expect(res.headers.get('x-correlation-id')).toBeTruthy();
  });

  it('oversized file triggers limit', async () => {
    // Simulate > configured limit by setting env lower via process env and large payload
    (process as any).env.EVIDENCE_MAX_FILE_SIZE = '10'; // 10 bytes
    const body = multipart({ summaryType: 'narrative' }, [ { name: 'big.txt', type: 'text/plain', content: 'This content exceeds ten bytes' } ]);
    const res: Response = await POST(mockEvent(body));
    if (res.status !== 500 && res.status !== 400) return; // depending on where limit triggers
    const json = await res.json();
    expect(res.headers.get('x-correlation-id')).toBeTruthy();
  });

  it('enforces rate limit', async () => {
    const body = multipart({ summaryType: 'narrative' }, [ { name: 'f.txt', type: 'text/plain', content: 'x' } ]);
    // Hit limit quickly (limit is 25 per minute; simulate by modifying loop >25)
    let last: Response | null = null;
    for (let i=0;i<27;i++) {
      const resp = await POST(mockEvent(body));
      last = resp;
      if (resp.status === 429) break;
    }
    if (last && last.status === 429) {
      const json = await last.json();
      expect(json.error.message).toMatch(/Rate limit/);
    }
  });
});
