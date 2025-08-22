// Prometheus-style metrics exposition for NLP embeddings.
// Integrates with nlpMetrics from sentence-transformer service.
import { nlpMetrics } from "./sentence-transformer";
// TODO: Fix import - // Orphaned content: import { getPipelineHistogram, getDedupeMetrics, getAutosolveMetrics, getQUICMetrics, getAggregateAnomaliesLast5m, getBudgetCounters import { getRedisMetrics } from './redis-metrics';

export function renderNlpMetrics(): string {
  const lines: string[] = [];
  // Basic counters and summaries (gauges) in Prometheus text format
  lines.push('# HELP nlp_embeddings_total Total number of embeddings computed');
  lines.push('# TYPE nlp_embeddings_total counter');
  lines.push(`nlp_embeddings_total ${nlpMetrics.embeddings_total}`);

  lines.push('# HELP nlp_embedding_cache_hits Cache hits for embeddings');
  lines.push('# TYPE nlp_embedding_cache_hits counter');
  lines.push(`nlp_embedding_cache_hits ${nlpMetrics.cache_hits}`);

  lines.push('# HELP nlp_embedding_cache_misses Cache misses for embeddings');
  lines.push('# TYPE nlp_embedding_cache_misses counter');
  lines.push(`nlp_embedding_cache_misses ${nlpMetrics.cache_misses}`);

  // Latency: export simple quantiles from collected window
  const lat = [...nlpMetrics.embed_latency_ms].sort((a,b)=>a-b);
  function pct(p:number){ if(lat.length===0) return 0; const idx = Math.min(lat.length-1, Math.floor(p*(lat.length-1))); return lat[idx]; }
  lines.push('# HELP nlp_embedding_latency_ms Approximate latency quantiles (summary)');
  lines.push('# TYPE nlp_embedding_latency_ms summary');
  lines.push(`nlp_embedding_latency_ms{quantile="0.5"} ${pct(0.5).toFixed(2)}`);
  lines.push(`nlp_embedding_latency_ms{quantile="0.9"} ${pct(0.9).toFixed(2)}`);
  lines.push(`nlp_embedding_latency_ms{quantile="0.99"} ${pct(0.99).toFixed(2)}`);
  const sum = lat.reduce((a,b)=>a+b,0);
  lines.push(`nlp_embedding_latency_ms_sum ${sum.toFixed(2)}`);
  lines.push(`nlp_embedding_latency_ms_count ${lat.length}`);

  lines.push('# HELP nlp_similarity_queries_total Similarity search invocations');
  lines.push('# TYPE nlp_similarity_queries_total counter');
  lines.push(`nlp_similarity_queries_total ${nlpMetrics.similarity_queries_total}`);

  // Embedding dedupe
  const dedupe = getDedupeMetrics();
  lines.push('# HELP nlp_embedding_dedupe_hits Embedding hash deduplication hits');
  lines.push('# TYPE nlp_embedding_dedupe_hits counter');
  lines.push(`nlp_embedding_dedupe_hits ${dedupe.hits}`);
  lines.push('# HELP nlp_embedding_dedupe_misses Embedding hash deduplication misses');
  lines.push('# TYPE nlp_embedding_dedupe_misses counter');
  lines.push(`nlp_embedding_dedupe_misses ${dedupe.misses}`);
  lines.push('# HELP nlp_embedding_dedupe_ratio Current dedupe hit ratio');
  lines.push('# TYPE nlp_embedding_dedupe_ratio gauge');
  lines.push(`nlp_embedding_dedupe_ratio ${dedupe.ratio.toFixed(4)}`);

  // Pipeline histogram
  const hists = getPipelineHistogram();
  lines.push('# HELP pipeline_stage_latency_ms Pipeline stage latency histogram (ms)');
  lines.push('# TYPE pipeline_stage_latency_ms histogram');
  for (const h of hists) {
    const { stage, buckets, counts, inf, sum: hsum, count, anomalies } = h as any;
    buckets.forEach((b, i) => lines.push(`pipeline_stage_latency_ms_bucket{stage="${stage}",le="${b}"} ${counts[i]}`));
    lines.push(`pipeline_stage_latency_ms_bucket{stage="${stage}",le="+Inf"} ${inf}`);
    lines.push(`pipeline_stage_latency_ms_sum{stage="${stage}"} ${hsum.toFixed(2)}`);
    lines.push(`pipeline_stage_latency_ms_count{stage="${stage}"} ${count}`);
    lines.push(`# HELP pipeline_stage_latency_anomalies_total Detected latency anomalies (negative or extreme outliers)`);
    lines.push(`# TYPE pipeline_stage_latency_anomalies_total counter`);
    lines.push(`pipeline_stage_latency_anomalies_total{stage="${stage}"} ${anomalies || 0}`);
  }

  // Autosolve summary
  const auto = getAutosolveMetrics();
  lines.push('# HELP autosolve_cycle_latency_ms Autosolve maintenance cycle latency summary (ms)');
  lines.push('# TYPE autosolve_cycle_latency_ms summary');
  lines.push(`autosolve_cycle_latency_ms{quantile="0.5"} ${auto.p50.toFixed(2)}`);
  lines.push(`autosolve_cycle_latency_ms{quantile="0.9"} ${auto.p90.toFixed(2)}`);
  lines.push(`autosolve_cycle_latency_ms{quantile="0.99"} ${auto.p99.toFixed(2)}`);
  lines.push(`autosolve_cycle_latency_ms_sum ${auto.sum.toFixed(2)}`);
  lines.push(`autosolve_cycle_latency_ms_count ${auto.count}`);

  // QUIC metrics
  const quic = getQUICMetrics();
  lines.push('# HELP quic_total_connections Current QUIC connections');
  lines.push('# TYPE quic_total_connections gauge');
  lines.push(`quic_total_connections ${quic.total_connections}`);
  lines.push('# HELP quic_total_streams Total QUIC streams opened');
  lines.push('# TYPE quic_total_streams counter');
  lines.push(`quic_total_streams ${quic.total_streams}`);
  lines.push('# HELP quic_total_errors Total QUIC errors');
  lines.push('# TYPE quic_total_errors counter');
  lines.push(`quic_total_errors ${quic.total_errors}`);
  lines.push('# HELP quic_avg_latency_ms Average QUIC stream latency (ms)');
  lines.push('# TYPE quic_avg_latency_ms gauge');
  lines.push(`quic_avg_latency_ms ${quic.avg_latency_ms}`);
  lines.push('# HELP quic_latency_ms QUIC latency quantiles');
  lines.push('# TYPE quic_latency_ms summary');
  lines.push(`quic_latency_ms{quantile="0.5"} ${quic.p50 || 0}`);
  lines.push(`quic_latency_ms{quantile="0.9"} ${quic.p90 || 0}`);
  lines.push(`quic_latency_ms{quantile="0.99"} ${quic.p99 || 0}`);
  lines.push('# HELP quic_error_events_last_minute QUIC errors in last 60s');
  lines.push('# TYPE quic_error_events_last_minute gauge');
  lines.push(`quic_error_events_last_minute ${quic.error_rate_1m || 0}`);
  // Aggregate anomalies
  lines.push('# HELP pipeline_latency_anomalies_last5m Total pipeline latency anomalies detected over last 5 minutes');
  lines.push('# TYPE pipeline_latency_anomalies_last5m gauge');
  lines.push(`pipeline_latency_anomalies_last5m ${getAggregateAnomaliesLast5m()}`);
  // Error budget counters
  const budgets = getBudgetCounters();
  lines.push('# HELP error_budget_quic_p99_breaches Total QUIC p99 latency budget breaches since start');
  lines.push('# TYPE error_budget_quic_p99_breaches counter');
  lines.push(`error_budget_quic_p99_breaches ${budgets.quicP99BudgetBreaches}`);
  lines.push('# HELP error_budget_pipeline_anomaly_spikes Total pipeline anomaly spike events since start');
  lines.push('# TYPE error_budget_pipeline_anomaly_spikes counter');
  lines.push(`error_budget_pipeline_anomaly_spikes ${budgets.pipelineAnomalySpikeBudget}`);

  // Redis metrics
  const redis = getRedisMetrics();
  lines.push('# HELP redis_up Redis health status (1=up,0=down)');
  lines.push('# TYPE redis_up gauge');
  lines.push(`redis_up ${redis.up}`);
  lines.push('# HELP redis_last_ping_ms Last successful Redis PING latency (ms)');
  lines.push('# TYPE redis_last_ping_ms gauge');
  lines.push(`redis_last_ping_ms ${redis.last_ping_ms}`);
  if (redis.last_ok_ts) {
    lines.push('# HELP redis_last_ok_timestamp_seconds Last time Redis ping succeeded (unix epoch)');
    lines.push('# TYPE redis_last_ok_timestamp_seconds gauge');
    lines.push(`redis_last_ok_timestamp_seconds ${(redis.last_ok_ts / 1000).toFixed(0)}`);
  }
  if (redis.last_error_ts) {
    lines.push('# HELP redis_last_error_timestamp_seconds Last time Redis ping failed (unix epoch)');
    lines.push('# TYPE redis_last_error_timestamp_seconds gauge');
    lines.push(`redis_last_error_timestamp_seconds ${(redis.last_error_ts / 1000).toFixed(0)}`);
  }

  return lines.join('\n') + '\n';
}
