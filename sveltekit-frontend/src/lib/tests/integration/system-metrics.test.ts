import { describe, it, expect } from 'vitest';
import { load as homeLoad } from '../../../routes/+page.server';
import { GET as metricsEndpoint } from '../../../routes/api/system/metrics/+server';

function createLoadEvent() {
  return {
    locals: {},
    fetch: async (_: string) => ({ ok: true, json: async () => ({}) }),
    setHeaders: (_: Record<string,string>) => {},
  } as any;
}

describe('System Metrics', () => {
  it('home page load returns metrics object', async () => {
    const data = await homeLoad(createLoadEvent());
    expect(data.metrics).toBeDefined();
    expect(typeof data.metrics.routingEfficiency).toBe('number');
    expect(typeof data.metrics.cacheHitRatio).toBe('number');
    expect(typeof data.metrics.gpuUtilization).toBe('number');
  });

  it('metrics endpoint returns success payload', async () => {
    const res = await metricsEndpoint({ locals: {}, request: new Request('http://localhost/api/system/metrics'), url: new URL('http://localhost/api/system/metrics') } as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.success).toBe(true);
    expect(body.data).toBeDefined();
    expect(typeof body.data.routingEfficiency).toBe('number');
  });
});
