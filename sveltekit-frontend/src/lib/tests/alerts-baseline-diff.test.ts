/// <reference types="vitest" />
// Tests baseline diff endpoint mode=baseline returns diff after first invocation

describe('Alerts Baseline Diff Endpoint', () => {
  let GET: any;
  let GET: any;
  beforeAll(async () => {
    const mod = await import('../../routes/api/v1/alerts/+server.js');
    GET = mod.GET;
  });

  async function call(mode?: string): Promise<any> {
    const url = new URL('http://localhost/api/v1/alerts' + (mode?`?mode=${mode}`:''));
    const res = await GET({ url } as any);
    const json = await res.json();
    return json;
  }

  it('returns baseline & null diff first call then diff second call', async () => {
    const first = await call('baseline');
    expect(first.baseline).toBeTruthy();
    expect(first.diff).toBeNull();
    // simulate minor metric accumulation by calling again quickly
    const second = await call('baseline');
    expect(second.baseline).toBeTruthy();
    expect(second.diff).not.toBeNull();
    expect(Array.isArray(second.diff.stageDiff)).toBe(true);
  });
});
