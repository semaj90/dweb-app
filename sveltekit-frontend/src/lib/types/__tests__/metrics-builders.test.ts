import { describe, it, expect } from 'vitest';
import { buildCognitiveMetrics, deriveEmergentCognitiveSignals } from '../metrics';

describe('Cognitive Metrics Builders', () => {
  it('buildCognitiveMetrics fills defaults and clamps', () => {
    const m = buildCognitiveMetrics({ routingEfficiency: 150, cacheHitRatio: -10, gpuUtilization: 42.5 });
    expect(m.routingEfficiency).toBe(100);
    expect(m.cacheHitRatio).toBe(0);
    expect(m.gpuUtilization).toBe(42.5);
    expect(typeof m.timestamp).toBe('string');
  });

  it('deriveEmergentCognitiveSignals adds emergent fields when absent', () => {
    const base = buildCognitiveMetrics({ routingEfficiency: 80, cacheHitRatio: 60, gpuUtilization: 50 });
    const derived = deriveEmergentCognitiveSignals(base);
    expect(derived.consciousnessLevel).toBeGreaterThan(0);
    expect(derived.quantumCoherence).toBeGreaterThan(0);
  });
});
