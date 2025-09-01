/// <reference types="vite/client" />
/// <reference types="vitest" />
import { vi } from 'vitest';

function buildRequest(body: any){
  return new Request('http://localhost/api/v1/quic/push', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  });
}

async function invokePOST(POST: any, body: any): Promise<any> {
  const req = buildRequest(body);
  const res = await POST({ request: req, getClientAddress: () => '127.0.0.1' } as any);
  const json = await res.json();
  return json;
}

describe('QUIC Push Alerts - sustained p99 & autosolve', () => {
  beforeAll(() => {
    import.meta.env.QUIC_ALERT_P99_MS = '100';
    import.meta.env.SUSTAINED_P99_THRESHOLD = '2';
  });

  it('increments sustained p99 breaches and triggers autosolve after threshold', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response('{}', { status: 200 })));
    global.fetch = fetchMock;

    const mod = await import('../../routes/api/v1/quic/push/+server.js');
    const POST = mod.POST;

    let r1 = await invokePOST(POST, {
      total_connections: 1, total_streams: 1, total_errors: 0,
      latencySamples: Array(20).fill(150)
    });
    expect(r1.ok).toBe(true);
    expect(r1.alerts).toContain('p99_latency_exceeded');
    expect(r1.sustainedP99.sustainedP99Breaches).toBe(1);
    expect(fetchMock).not.toHaveBeenCalled();

    let r2 = await invokePOST(POST, {
      total_connections: 1, total_streams: 2, total_errors: 0,
      latencySamples: Array(10).fill(160)
    });
    expect(r2.alerts).toContain('p99_latency_exceeded');
    expect(r2.sustainedP99.sustainedP99Breaches).toBe(2);
    await new Promise(r => setTimeout(r, 10));
    expect(fetchMock).toHaveBeenCalled();
    const autosolveCall = fetchMock.mock.calls.find((c: any[]) => c.length > 0 && c[0] && (c[0] as string).includes('context7-autosolve'));
    expect(autosolveCall).toBeTruthy();

    import.meta.env.QUIC_ALERT_P99_MS = '500';
    const r3 = await invokePOST(POST, {
      total_connections: 1, total_streams: 3, total_errors: 0,
      latencySamples: Array(5).fill(120)
    });
    expect(r3.alerts.length === 0 || !r3.alerts.includes('p99_latency_exceeded')).toBe(true);
    expect(r3.sustainedP99.sustainedP99Breaches).toBe(0);
  });
});
