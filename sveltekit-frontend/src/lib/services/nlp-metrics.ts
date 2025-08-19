// Prometheus-style metrics exposition for NLP embeddings.
// Integrates with nlpMetrics from sentence-transformer service.
import { nlpMetrics } from './sentence-transformer';

export function renderNlpMetrics(): string {
  // Basic counters and summaries (gauges) in Prometheus text format
  const lines: string[] = [];
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

  return lines.join('\n') + '\n';
}
