import { describe, it, expect } from "vitest";
// Orphaned content: import {
recordStageLatency, getPipelineHistogram, updateQUICMetrics, getQUICMetrics

describe('pipeline anomaly logic', () => {
  it('flags large outlier as anomaly', () => {
    for(let i=0;i<20;i++) recordStageLatency('gpu', 10);
    recordStageLatency('gpu', 200); // 20x median -> anomaly
    const hist = getPipelineHistogram();
    const gpu = hist.find(h=>h.stage==='gpu') as any;
    expect(gpu.anomalies).toBeGreaterThan(0);
  });
});

describe('QUIC quantiles', () => {
  it('computes p50/p90/p99', () => {
    [5,10,15,20,25,30,100].forEach(v=> updateQUICMetrics({ latencySample: v }));
    const q = getQUICMetrics();
    expect(q.p50).toBeGreaterThan(0);
    expect(q.p99).toBeGreaterThanOrEqual(q.p90);
  });
});
