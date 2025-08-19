// In-process metrics aggregation for pipeline stages, dedupe, QUIC, autosolve.
// Lightweight; updated by client or future server push adapters.

export type PipelineStage = 'gpu' | 'wasm' | 'embedding' | 'retrieval' | 'llm' | 'final';

interface StageStats { samples: number[]; sum: number; count: number; anomalies?: number; anomalyTimestamps?: number[] }
const STAGES: PipelineStage[] = ['gpu','wasm','embedding','retrieval','llm','final'];
const stageData: Record<PipelineStage, StageStats> = STAGES.reduce((acc, s) => { acc[s] = { samples: [], sum: 0, count: 0, anomalies: 0, anomalyTimestamps: [] }; return acc; }, {} as any);

let dedupeHits = 0; let dedupeMisses = 0;
const autosolveDurations: number[] = [];
interface QuicMetricsShape { total_connections: number; total_streams: number; total_errors: number; avg_latency_ms: number; timestamp: number; latencies: number[]; errorsWindow: number[] }
let quicMetrics: QuicMetricsShape = { total_connections: 0, total_streams: 0, total_errors: 0, avg_latency_ms: 0, timestamp: Date.now(), latencies: [], errorsWindow: [] };
// Error budget accounting (simple in-memory counters resettable via process restart)
let quicP99BudgetBreaches = 0; // increments when p99 above threshold during update call (optional external call site)
let pipelineAnomalySpikeBudget = 0; // increments when anomaly spike detected externally
export function noteQuicP99Breach() { quicP99BudgetBreaches++; }
export function notePipelineAnomalySpike() { pipelineAnomalySpikeBudget++; }
export function getBudgetCounters() { return { quicP99BudgetBreaches, pipelineAnomalySpikeBudget }; }
export function resetBudgetCounters() { quicP99BudgetBreaches = 0; pipelineAnomalySpikeBudget = 0; }

export function recordStageLatency(stage: PipelineStage, ms: number) {
    const s = stageData[stage];
    const now = Date.now();
    const markAnomaly = () => {
        s.anomalies = (s.anomalies || 0) + 1;
        s.anomalyTimestamps?.push(now);
        // prune >5m
        const cutoff = now - 5 * 60 * 1000;
        s.anomalyTimestamps = (s.anomalyTimestamps || []).filter(t => t >= cutoff);
    };
    if (ms < 0) { markAnomaly(); return; }
    const sorted = [...s.samples].sort((a, b) => a - b);
    const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : ms;
    if (sorted.length && ms > 5 * (median || 1)) { markAnomaly(); }
    s.samples.push(ms);
    if (s.samples.length > 500) s.samples.shift();
    s.sum += ms; s.count++;
}
export function recordEmbeddingDedupe(hit:boolean){ if(hit) dedupeHits++; else dedupeMisses++; }
export function recordAutosolveCycle(ms:number){ autosolveDurations.push(ms); if(autosolveDurations.length>200) autosolveDurations.shift(); }
export function updateQUICMetrics(m: Partial<QuicMetricsShape> & { latencySample?: number; errorOccurred?: boolean }) {
    const now = Date.now();
    if (m.latencySample !== undefined) {
        quicMetrics.latencies.push(m.latencySample);
        if (quicMetrics.latencies.length > 300) quicMetrics.latencies.shift();
    }
    if (m.errorOccurred) {
        quicMetrics.errorsWindow.push(now);
        const cutoff = now - 5 * 60 * 1000;
        quicMetrics.errorsWindow = quicMetrics.errorsWindow.filter(ts => ts >= cutoff);
    }
    quicMetrics = { ...quicMetrics, ...m, timestamp: now };
    if (quicMetrics.latencies.length) {
        const sum = quicMetrics.latencies.reduce((a, b) => a + b, 0);
        quicMetrics.avg_latency_ms = parseFloat((sum / quicMetrics.latencies.length).toFixed(2));
    }
}

export function getPipelineHistogram() {
    const buckets = [5, 10, 20, 50, 100, 200, 500, 1000, 2000];
    return STAGES.map(stage => {
        const { samples, sum, count, anomalies } = stageData[stage];
        const counts = buckets.map(b => samples.filter(v => v <= b).length);
        const inf = samples.length;
        return { stage, buckets, counts, inf, sum, count, anomalies: anomalies || 0, recentSamples: samples.slice(-25) };
    });
}
export function getDedupeMetrics(){ const total=dedupeHits+dedupeMisses; return { hits:dedupeHits, misses:dedupeMisses, ratio: total? dedupeHits/total:0 }; }
export function getAutosolveMetrics(){ const arr=[...autosolveDurations].sort((a,b)=>a-b); const count=arr.length; const sum=arr.reduce((a,b)=>a+b,0); const p=(q:number)=> count? arr[Math.min(count-1, Math.floor(q*(count-1)))] : 0; return { count,sum,p50:p(0.5),p90:p(0.9),p99:p(0.99) }; }
export function getQUICMetrics() {
    const arr = [...quicMetrics.latencies].sort((a, b) => a - b);
    const q = (p: number) => arr.length ? arr[Math.min(arr.length - 1, Math.floor(p * (arr.length - 1)))] : 0;
    const now = Date.now();
    const oneMinCut = now - 60_000;
    const errors1m = quicMetrics.errorsWindow.filter(ts => ts >= oneMinCut).length;
    return { ...quicMetrics, p50: q(0.5), p90: q(0.9), p99: q(0.99), error_rate_1m: errors1m };
}

export function getAggregateAnomaliesLast5m() {
    const cutoff = Date.now() - 5 * 60 * 1000;
    let total = 0;
    for (const st of STAGES) {
        total += (stageData[st].anomalyTimestamps || []).filter(t => t >= cutoff).length;
    }
    return total;
}

// Baseline helpers -----------------------------------------------------------
export interface StageBaselineEntry { stage: PipelineStage; p50: number; p90: number; p99: number; count: number; anomalies: number }

export function getStageBaselineSnapshot(): StageBaselineEntry[] {
    return STAGES.map(stage => {
        const s = stageData[stage];
        const arr = [...s.samples].sort((a, b) => a - b);
        const q = (p: number) => arr.length ? arr[Math.min(arr.length - 1, Math.floor(p * (arr.length - 1)))] : 0;
        return { stage, p50: q(0.5), p90: q(0.9), p99: q(0.99), count: s.count, anomalies: s.anomalies || 0 };
    });
}

