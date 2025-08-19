// Central lightweight alert routing & history ring buffer
import { getQUICMetrics, getAggregateAnomaliesLast5m, getStageBaselineSnapshot, resetBudgetCounters, getBudgetCounters } from "./pipeline-metrics.js";
import { natsMessaging, import fs from 'fs';, import path from 'path';, // Frontend NATS subject for system alerts (mirrors SYSTEM_ALERTS pattern in backend service), const SYSTEM_ALERTS_SUBJECT = 'system.alerts';, , export interface RuntimeAlert {,   id: string;,   type: string; // e.g., p99_latency_exceeded, error_spike, pipeline_anomaly_spike,   severity: 'info' | 'warn' | 'critical';,   message: string;,   ts: number;,   data?: Record<string, any>; } from

const RING_SIZE = 200;
const alerts: RuntimeAlert[] = [];
let sustainedP99Breaches = 0; // consecutive cycles containing p99 breach alerts
let lastP99Ok = Date.now();
const SUSTAINED_P99_THRESHOLD = Number(import.meta.env.SUSTAINED_P99_THRESHOLD || 3);

// ---------------------------------------------------------------------------
// Persistence of observability state (sustained counters, last baseline, budgets)
// ---------------------------------------------------------------------------
const RUNTIME_DIR = path.resolve(process.cwd(), '.runtime');
const STATE_FILE = path.join(RUNTIME_DIR, 'observability-state.json');
interface BaselineFile { created: string; stages: ReturnType<typeof getStageBaselineSnapshot>; quic: any }
interface PersistedState { sustainedP99Breaches:number; lastP99Ok:number; lastBaseline?: BaselineFile; budgets?: any; savedAt: string }
let lastPersistBaseline: BaselineFile | undefined;
(function loadPersisted(){
  try {
    if(fs.existsSync(STATE_FILE)){
      const raw = JSON.parse(fs.readFileSync(STATE_FILE,'utf8')) as PersistedState;
      sustainedP99Breaches = raw.sustainedP99Breaches || 0;
      lastP99Ok = raw.lastP99Ok || Date.now();
      if(raw.lastBaseline) lastPersistBaseline = raw.lastBaseline;
    }
  } catch(err){
    console.warn('Observability state load failed:', (err as any).message);
  }
})();
function persistState(){
  try {
    if(!fs.existsSync(RUNTIME_DIR)) fs.mkdirSync(RUNTIME_DIR,{recursive:true});
    const payload: PersistedState = { sustainedP99Breaches, lastP99Ok, lastBaseline: lastPersistBaseline, budgets: getBudgetCounters(), savedAt: new Date().toISOString() };
    fs.writeFileSync(STATE_FILE, JSON.stringify(payload,null,2));
  } catch(err){
    console.warn('Observability state persist failed:', (err as any).message);
  }
}
setInterval(persistState, 60_000).unref?.();

export function routeAlerts(raw: string[], ctx: Record<string, any>){
  if(!raw.length) { sustainedP99Breaches = 0; return [] as RuntimeAlert[]; }
  const quic = getQUICMetrics();
  const out: RuntimeAlert[] = [];
  for(const code of raw){
    let severity: RuntimeAlert['severity'] = 'info';
    if(code === 'p99_latency_exceeded') severity = 'warn';
    if(code === 'error_spike' || code === 'pipeline_anomaly_spike') severity = 'critical';
    const alert: RuntimeAlert = { id: `${Date.now()}-${Math.random().toString(36).slice(2,8)}`, type: code, severity, message: humanize(code, quic), ts: Date.now(), data: { quicP99: quic.p99, quicErrors1m: quic.error_rate_1m, anomalies5m: getAggregateAnomaliesLast5m(), ...(ctx||{}) } };
    pushAlert(alert); out.push(alert);
  }
  if(raw.includes('p99_latency_exceeded')) { sustainedP99Breaches++; } else { if(sustainedP99Breaches>0) lastP99Ok = Date.now(); sustainedP99Breaches = 0; }
  return out;
}

function pushAlert(a: RuntimeAlert){
  alerts.push(a); if(alerts.length > RING_SIZE) alerts.shift();
  // Simple console log routing; NATS or other transports can hook here
  // eslint-disable-next-line no-console
  console.log(`[ALERT][${a.severity}] ${a.type} :: ${a.message}`);
  try { if(natsMessaging && (natsMessaging as any).publish){ (natsMessaging as any).publish(SYSTEM_ALERTS_SUBJECT, { id: a.id, type: a.type, severity: a.severity, message: a.message, ts: a.ts, data: a.data }); } } catch(err){ console.warn('NATS alert publish failed', (err as any)?.message); }
}

function humanize(code: string, quic: any){
  switch(code){
    case 'p99_latency_exceeded': return `QUIC p99 ${quic.p99}ms exceeded threshold`;
    case 'error_spike': return `QUIC error spike (last min ${quic.error_rate_1m})`;
    case 'pipeline_anomaly_spike': return `Pipeline anomaly spike (${getAggregateAnomaliesLast5m()} anomalies in 5m)`;
    default: return code;
  }
}

export function getAlertHistory(){ return [...alerts].sort((a,b)=> b.ts - a.ts); }

// Autosolve trigger logic --------------------------------------------------
let autosolveInFlight = false; let lastAutosolveTs = 0; const AUTOSOLVE_COOLDOWN_MS = 5 * 60 * 1000;
export async function maybeTriggerAutosolve(fetchFn: typeof fetch, rawCodes: string[]){
  const need = shouldAutosolve(rawCodes); if(!need) return { triggered:false }; if(autosolveInFlight) return { triggered:false, reason:'in_flight' }; if(Date.now() - lastAutosolveTs < AUTOSOLVE_COOLDOWN_MS) return { triggered:false, reason:'cooldown' };
  autosolveInFlight = true; try { const start = performance.now(); const resp = await fetchFn('/api/context7-autosolve?action=trigger',{method:'POST'}); const dur = performance.now()-start; lastAutosolveTs = Date.now(); return { triggered:true, status: resp.status, durationMs: dur }; } catch(e:any){ return { triggered:false, error: e.message }; } finally { autosolveInFlight = false; }
}
function shouldAutosolve(codes: string[]){ if(codes.includes('pipeline_anomaly_spike')) return true; if(codes.includes('p99_latency_exceeded') && sustainedP99Breaches >= SUSTAINED_P99_THRESHOLD) return true; return false; }

export function getSustainedP99Info(){ return { sustainedP99Breaches, threshold: SUSTAINED_P99_THRESHOLD, lastP99OkTs: lastP99Ok }; }

// Baseline diff ------------------------------------------------------------
export function buildBaseline(): BaselineFile { const b = { created: new Date().toISOString(), stages: getStageBaselineSnapshot(), quic: getQUICMetrics() }; lastPersistBaseline = b; persistState(); return b; }
export function diffBaselines(oldB: BaselineFile, newB: BaselineFile){ const stageDiff = newB.stages.map(s=>{ const prev = oldB.stages.find(p=>p.stage===s.stage); if(!prev) return { stage:s.stage, change:'added', current:s }; const deltas = { p50: s.p50 - prev.p50, p90: s.p90 - prev.p90, p99: s.p99 - prev.p99, anomalies: s.anomalies - prev.anomalies }; return { stage:s.stage, deltas }; }); return { stageDiff, quicP99Delta: newB.quic.p99 - (oldB.quic.p99||0) }; }

// Daily reset scheduler ----------------------------------------------------
const DAILY_RESET_HOUR = Number(import.meta.env.OBS_DAILY_RESET_HOUR || 0); // UTC hour
function msUntilNextReset(){ const now = new Date(); const next = new Date(now.getTime()); next.setUTCHours(DAILY_RESET_HOUR,0,0,0); if(next <= now) next.setUTCDate(next.getUTCDate()+1); return next.getTime() - now.getTime(); }
(function scheduleDailyReset(){ setTimeout(()=>{ resetBudgetCounters(); sustainedP99Breaches = 0; lastP99Ok = Date.now(); persistState(); scheduleDailyReset(); }, msUntilNextReset()).unref?.(); })();
