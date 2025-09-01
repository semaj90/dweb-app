import { updateQUICMetrics, getQUICMetrics, getAggregateAnomaliesLast5m, noteQuicP99Breach, notePipelineAnomalySpike } from "drizzle-orm";
import { routeAlerts, maybeTriggerAutosolve, getSustainedP99Info } from '$lib/services/alert-center';
import type { RequestHandler } from './$types';


// Simple in-memory rate limit: 60 requests per minute per IP
const hits: Record<string, number[]> = {};

const MAX_LAT_SAMPLES = 200; // guard
export const POST: RequestHandler = async ({ request, getClientAddress }) => {
  try {
    const ip = getClientAddress();
    const now = Date.now();
    hits[ip] = (hits[ip]||[]).filter(ts => ts > now - 60_000);
    if (hits[ip].length >= 60) {
      return new Response(JSON.stringify({ ok:false, error:'rate_limited' }), { status:429, headers:{'content-type':'application/json'} });
    }
    hits[ip].push(now);
    const body = await request.json();
    if (body && typeof body === 'object') {
      if (body.latencySamples && !Array.isArray(body.latencySamples)) throw new Error('latencySamples must be array');
      if (Array.isArray(body.latencySamples) && body.latencySamples.length > MAX_LAT_SAMPLES) throw new Error('too_many_latency_samples');
      if (Array.isArray(body.latencySamples)) {
        for(const s of body.latencySamples){ if (typeof s === 'number' && s >=0 && s < 120000) updateQUICMetrics({ latencySample: s }); }
      }
    }
    updateQUICMetrics({
      total_connections: body.total_connections,
      total_streams: body.total_streams,
      total_errors: body.total_errors,
      errorOccurred: body.errorOccurred
    });
    // Alert threshold evaluation
    const quic = getQUICMetrics();
    const alerts: string[] = [];
    const p99Threshold = Number(import.meta.env.QUIC_ALERT_P99_MS||800);
    const err1mThreshold = Number(import.meta.env.QUIC_ALERT_ERRORS_1M||5);
  if (quic.p99 && quic.p99 > p99Threshold) { alerts.push('p99_latency_exceeded'); noteQuicP99Breach(); }
    if ((quic.error_rate_1m||0) > err1mThreshold) alerts.push('error_spike');
    const anomalies5m = getAggregateAnomaliesLast5m();
    const anomalyThreshold = Number(import.meta.env.PIPELINE_ALERT_ANOMALIES_5M||20);
  if (anomalies5m > anomalyThreshold) { alerts.push('pipeline_anomaly_spike'); notePipelineAnomalySpike(); }
    // Route alerts (history + console) and maybe autosolve
    const routed = routeAlerts(alerts, { source:'quic_push' });
    if(alerts.length){
      // fire-and-forget autosolve trigger (no await blocking response)
      maybeTriggerAutosolve(fetch, alerts).catch(()=>{});
    }
  const sustained = getSustainedP99Info();
  return new Response(JSON.stringify({ ok:true, alerts, routedCount: routed.length, p99: quic.p99, errors_1m: quic.error_rate_1m, anomalies5m, sustainedP99: sustained }), { status:200, headers:{'content-type':'application/json'} });
  } catch (e:any){
    return new Response(JSON.stringify({ ok:false, error: e.message }), { status:400, headers:{'content-type':'application/json'} });
  }
};
