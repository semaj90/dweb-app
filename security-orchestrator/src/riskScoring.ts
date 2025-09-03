import { SecurityValidationRequest } from './schemas';
import { randomUUID } from 'node:crypto';

export interface SignalResult {
  name: string;
  weight: number; // 0-1 relative importance
  value: any;
  contribution: number; // signed value added to risk baseline
  debug?: string;
}

export interface RiskComputationResult {
  requestId: string;
  riskScore: number;
  securityScore: number;
  signals: SignalResult[];
  status: 'allow' | 'review' | 'deny';
  modelVersion: string;
  durationMs: number;
}

const MODEL_VERSION = 'heuristic-v0.1';

export function computeRisk(input: SecurityValidationRequest): RiskComputationResult {
  const start = performance.now();
  const signals: SignalResult[] = [];

  function addSignal(sig: SignalResult) { signals.push(sig); }

  // Baseline risk
  let risk = 0.15; // neutral baseline

  const { fingerprint, user, context } = input;

  // User agent entropy heuristic
  if (fingerprint.userAgent) {
    const entropy = Math.min(1, fingerprint.userAgent.length / 120);
    const contribution = (entropy < 0.3 ? 0.05 : -0.02);
    addSignal({ name: 'userAgentEntropy', weight: 0.4, value: entropy, contribution, debug: 'Short UA may indicate automation' });
    risk += contribution;
  }

  // GPU / renderer presence (headless envs often generic)
  if (fingerprint.webglRenderer) {
    const generic = /swiftshader|llvmpipe|mesa/i.test(fingerprint.webglRenderer);
    const contribution = generic ? 0.07 : -0.03;
    addSignal({ name: 'webglRendererGeneric', weight: 0.3, value: fingerprint.webglRenderer, contribution, debug: 'Generic renderer raises risk' });
    risk += contribution;
  }

  // Hardware concurrency
  if (fingerprint.hardwareConcurrency) {
    const hc = fingerprint.hardwareConcurrency;
    let contribution = 0;
    if (hc <= 2) contribution = 0.04; // cheap bot infra
    else if (hc >= 16) contribution = 0.03; // suspiciously large (cloud?)
    else contribution = -0.02; // normal consumer device
    addSignal({ name: 'hardwareConcurrency', weight: 0.25, value: hc, contribution });
    risk += contribution;
  }

  // Behavioral signals
  const beh = user.behavioral;
  if (beh) {
    if (typeof beh.mouseMovementEntropy === 'number') {
      const ent = beh.mouseMovementEntropy;
      const contribution = ent < 0.2 ? 0.06 : ent < 0.4 ? 0.03 : -0.02;
      addSignal({ name: 'mouseMovementEntropy', weight: 0.5, value: ent, contribution });
      risk += contribution;
    }
    if (typeof beh.keypressVariance === 'number') {
      const kv = beh.keypressVariance;
      const contribution = kv < 0.15 ? 0.05 : kv < 0.35 ? 0.02 : -0.02;
      addSignal({ name: 'keypressVariance', weight: 0.45, value: kv, contribution });
      risk += contribution;
    }
    if (typeof beh.interactionLatencyMs === 'number') {
      const lat = beh.interactionLatencyMs;
      const contribution = lat < 40 ? 0.05 : lat > 5000 ? 0.02 : -0.01;
      addSignal({ name: 'interactionLatencyMs', weight: 0.2, value: lat, contribution });
      risk += contribution;
    }
  }

  // Context velocity (rapid attempts)
  if (context?.velocity !== undefined) {
    const v = context.velocity;
    const contribution = v > 0.7 ? 0.09 : v > 0.4 ? 0.04 : -0.01;
    addSignal({ name: 'velocity', weight: 0.5, value: v, contribution });
    risk += contribution;
  }

  // Previous failures
  if (context?.previousFailures) {
    const pf = context.previousFailures;
    const contribution = Math.min(0.15, pf * 0.03);
    addSignal({ name: 'previousFailures', weight: 0.35, value: pf, contribution });
    risk += contribution;
  }

  // IP reputation
  if (context?.ipReputation !== undefined) {
    const rep = context.ipReputation; // 0 bad → 1 good
    const contribution = rep < 0.2 ? 0.15 : rep < 0.4 ? 0.07 : rep < 0.6 ? 0.03 : -0.04;
    addSignal({ name: 'ipReputation', weight: 0.6, value: rep, contribution });
    risk += contribution;
  }

  // Clamp risk
  risk = Math.min(1, Math.max(0, risk));

  // Security score inverse (with mild smoothing)
  const securityScore = Math.min(1, Math.max(0, 1 - (risk * 0.92)));

  // Decision thresholds
  let status: 'allow' | 'review' | 'deny';
  if (risk < 0.35) status = 'allow';
  else if (risk < 0.65) status = 'review';
  else status = 'deny';

  const durationMs = Math.round(performance.now() - start);
  return {
    requestId: randomUUID(),
    riskScore: Number(risk.toFixed(4)),
    securityScore: Number(securityScore.toFixed(4)),
    signals: signals.sort((a,b) => Math.abs(b.weight * b.contribution) - Math.abs(a.weight * a.contribution)).slice(0, 25),
    status,
    modelVersion: MODEL_VERSION,
    durationMs
  };
}
