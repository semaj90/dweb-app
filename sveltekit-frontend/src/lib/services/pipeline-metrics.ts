// In-process metrics aggregation for pipeline stages, dedupe, QUIC, autosolve.
// Lightweight; updated by client or future server push adapters.

export type PipelineStage = 'gpu' | 'wasm' | 'embedding' | 'retrieval' | 'llm' | 'final';

interface StageStats { samples: number[]; sum: number; count: number }
const STAGES: PipelineStage[] = ['gpu','wasm','embedding','retrieval','llm','final'];
const stageData: Record<PipelineStage, StageStats> = STAGES.reduce((acc, s) => { acc[s] = { samples: [], sum:0, count:0 }; return acc; }, {} as any);

let dedupeHits = 0; let dedupeMisses = 0;
const autosolveDurations: number[] = [];
let quicMetrics = { total_connections:0,total_streams:0,total_errors:0,avg_latency_ms:0,timestamp:Date.now() };

export function recordStageLatency(stage: PipelineStage, ms: number){ const s=stageData[stage]; s.samples.push(ms); if(s.samples.length>500) s.samples.shift(); s.sum+=ms; s.count++; }
export function recordEmbeddingDedupe(hit:boolean){ if(hit) dedupeHits++; else dedupeMisses++; }
export function recordAutosolveCycle(ms:number){ autosolveDurations.push(ms); if(autosolveDurations.length>200) autosolveDurations.shift(); }
export function updateQUICMetrics(m: Partial<typeof quicMetrics>){ quicMetrics = { ...quicMetrics, ...m, timestamp: Date.now() }; }

export function getPipelineHistogram(){ const buckets=[5,10,20,50,100,200,500,1000,2000]; return STAGES.map(stage=>{ const {samples,sum,count}=stageData[stage]; const counts=buckets.map(b=>samples.filter(v=>v<=b).length); const inf=samples.length; return {stage,buckets,counts,inf,sum,count}; }); }
export function getDedupeMetrics(){ const total=dedupeHits+dedupeMisses; return { hits:dedupeHits, misses:dedupeMisses, ratio: total? dedupeHits/total:0 }; }
export function getAutosolveMetrics(){ const arr=[...autosolveDurations].sort((a,b)=>a-b); const count=arr.length; const sum=arr.reduce((a,b)=>a+b,0); const p=(q:number)=> count? arr[Math.min(count-1, Math.floor(q*(count-1)))] : 0; return { count,sum,p50:p(0.5),p90:p(0.9),p99:p(0.99) }; }
export function getQUICMetrics(){ return quicMetrics; }
