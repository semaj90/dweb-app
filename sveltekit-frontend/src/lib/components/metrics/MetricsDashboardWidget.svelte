<script lang="ts">
  import { onMount } from 'svelte';
  type NLPQuantiles = { p50:number; p90:number; p99:number };
  interface NLPStats { embeddings_total:number; cache_hits:number; cache_misses:number; latency: NLPQuantiles; similarity_queries_total:number; hit_ratio:number; dedupe_hits?:number; dedupe_misses?:number; dedupe_ratio?:number }
  interface NATSMetricSnapshot { connection:{ status:string; since:number|null; reconnectAttempts:number }; messaging:{ published:number; received:number; subjects: Record<string,string[]> } }
  interface PipelineHistogram { stage:string; buckets:number[]; counts:number[]; inf:number; sum:number; count:number; recentSamples?: number[] }
  interface AutosolveMetrics { count:number; sum:number; p50:number; p90:number; p99:number }
  interface QUICMetrics { total_connections:number; total_streams:number; total_errors:number; avg_latency_ms:number }
  interface RedisMetrics { up:number; last_ping_ms:number; last_ok_ts?:number|null; last_error_ts?:number|null }
  let loading = true; let error: string | null = null; let nlp: NLPStats | null = null; let nats: NATSMetricSnapshot | null = null; let pipeline: PipelineHistogram[] = []; let autosolve: AutosolveMetrics | null = null; let quic: QUICMetrics | null = null; let redis: RedisMetrics | null = null; let lastRefreshed: Date | null = null;
  async function fetchNLP(){
    // Prometheus text parsing from /api/v1/nlp/metrics
    const res = await fetch('/api/v1/nlp/metrics');
    if(!res.ok) throw new Error('NLP metrics fetch failed');
    const body = await res.text();
    const lines = body.split(/\n+/);
    const kv: Record<string, number> = {};
    for(const ln of lines){
      if(!ln || ln.startsWith('#')) continue;
      const parts = ln.trim().split(/\s+/);
      if(parts.length>=2){
        const key = parts[0];
        const val = parseFloat(parts[1]);
        if(!Number.isNaN(val)) kv[key]=val;
      }
    }
    const latency: NLPQuantiles = { p50: kv['nlp_embedding_latency_ms{quantile="0.5"}']||0, p90: kv['nlp_embedding_latency_ms{quantile="0.9"}']||0, p99: kv['nlp_embedding_latency_ms{quantile="0.99"}']||0 };
    const embeddings_total = kv['nlp_embeddings_total']||0;
    const cache_hits = kv['nlp_embedding_cache_hits']||0;
    const cache_misses = kv['nlp_embedding_cache_misses']||0;
    const similarity_queries_total = kv['nlp_similarity_queries_total']||0;
    const hit_ratio = cache_hits + cache_misses > 0 ? cache_hits/(cache_hits+cache_misses) : 0;
    return { embeddings_total, cache_hits, cache_misses, similarity_queries_total, latency, hit_ratio } as NLPStats;
  }

  async function fetchNATS(){
    const res = await fetch('/api/v1/nats/metrics');
    if(!res.ok) throw new Error('NATS metrics fetch failed');
    const js = await res.json();
    return js.metrics as NATSMetricSnapshot;
  }

  async function fetchPipeline(){
    const res = await fetch('/api/v1/pipeline/metrics');
    if(!res.ok) throw new Error('Pipeline metrics fetch failed');
    const js = await res.json();
    pipeline = js.pipeline || [];
    if (js.dedupe && nlp) {
      nlp.dedupe_hits = js.dedupe.hits;
      nlp.dedupe_misses = js.dedupe.misses;
      nlp.dedupe_ratio = js.dedupe.ratio;
    }
  }

  async function fetchAutosolve(){
    const res = await fetch('/api/v1/autosolve/metrics');
    if(!res.ok) throw new Error('Autosolve metrics fetch failed');
    const js = await res.json();
    autosolve = js.autosolve;
  }

  async function fetchQUIC(){
    const res = await fetch('/api/v1/quic/metrics');
    if(!res.ok) throw new Error('QUIC metrics fetch failed');
    const js = await res.json();
    quic = js.quic;
  }

  function loadPersisted(){
    if (typeof localStorage === 'undefined') return;
    try {
      const raw = localStorage.getItem('metricsDashboardCache');
      if (!raw) return;
      const parsed = JSON.parse(raw);
      nlp = parsed.nlp || nlp;
      nats = parsed.nats || nats;
      pipeline = parsed.pipeline || pipeline;
      autosolve = parsed.autosolve || autosolve;
      quic = parsed.quic || quic;
      redis = parsed.redis || redis;
      lastRefreshed = parsed.lastRefreshed ? new Date(parsed.lastRefreshed) : lastRefreshed;
    } catch {}
  }

  function persist(){
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem('metricsDashboardCache', JSON.stringify({ nlp, nats, pipeline, autosolve, quic, redis, lastRefreshed }));
    } catch {}
  }

  async function refresh(){
    loading = true; error = null;
    try {
  const [nlpRes, natsRes, pipeRes, autoRes, quicRes, redisRes] = await Promise.allSettled([fetchNLP(), fetchNATS(), fetchPipeline(), fetchAutosolve(), fetchQUIC(), fetch('/api/v1/redis/metrics').then(r=> r.ok? r.json(): Promise.reject(new Error('Redis metrics fetch failed')))]);
      if(nlpRes.status==='fulfilled') nlp = nlpRes.value; else error = (nlpRes.reason?.message)||error;
      if(natsRes.status==='fulfilled') nats = natsRes.value; else error = (natsRes.reason?.message)||error;
      if(pipeRes.status==='rejected') error = (pipeRes.reason?.message)||error;
      if(autoRes.status==='rejected') error = (autoRes.reason?.message)||error;
      if(quicRes.status==='rejected') error = (quicRes.reason?.message)||error;
  if(redisRes.status==='fulfilled') redis = (redisRes.value as any).redis; else error = (redisRes.reason?.message)||error;
      lastRefreshed = new Date();
      persist();
    } catch (e:any){ error = e.message; }
    loading = false;
  }

  let interval: any;
  onMount(()=>{ loadPersisted(); refresh(); interval = setInterval(()=>{
    // If QUIC metrics stale (>30s) trigger only QUIC refresh more frequently
    if (quic && Date.now() - (lastRefreshed?.getTime()||0) > 30000) { fetchQUIC().catch(()=>{}); }
    refresh();
  }, 10000); return ()=> clearInterval(interval); });
</script>

<div class="metrics-widget border rounded-lg p-4 bg-slate-900 text-slate-100 text-sm space-y-4">
  <div class="flex items-center justify-between">
    <h2 class="text-base font-semibold">AI Metrics Dashboard</h2>
  <button class="px-2 py-1 text-xs rounded bg-slate-700 hover:bg-slate-600" onclick={refresh} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>
  </div>
  {#if error}
    <div class="text-red-400">{error}</div>
  {/if}
  <div class="grid md:grid-cols-4 gap-4">
    <section class="space-y-2">
      <h3 class="font-medium text-indigo-300">NLP Embeddings</h3>
      {#if nlp}
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
          <span class="opacity-70">Total</span><span>{nlp.embeddings_total}</span>
          <span class="opacity-70">Cache Hits</span><span>{nlp.cache_hits}</span>
          <span class="opacity-70">Cache Misses</span><span>{nlp.cache_misses}</span>
          <span class="opacity-70">Hit Ratio</span><span>{(nlp.hit_ratio*100).toFixed(1)}%</span>
          <span class="opacity-70">Similarity Q</span><span>{nlp.similarity_queries_total}</span>
          <span class="opacity-70">Latency p50</span><span>{nlp.latency.p50} ms</span>
          <span class="opacity-70">Latency p90</span><span>{nlp.latency.p90} ms</span>
            <span class="opacity-70">Latency p99</span><span>{nlp.latency.p99} ms</span>
          <span class="opacity-70">Dedupe Hits</span><span>{nlp.dedupe_hits ?? 0}</span>
          <span class="opacity-70">Dedupe Misses</span><span>{nlp.dedupe_misses ?? 0}</span>
          <span class="opacity-70">Dedupe Ratio</span><span>{((nlp.dedupe_ratio||0)*100).toFixed(1)}%</span>
        </div>
      {:else}<div class="opacity-70">No NLP data</div>{/if}
    </section>
    <section class="space-y-2">
      <h3 class="font-medium text-emerald-300">NATS Messaging</h3>
      {#if nats}
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
          <span class="opacity-70">Status</span><span class={nats.connection.status==='connected' ? 'text-green-400':'text-red-400'}>{nats.connection.status}</span>
          <span class="opacity-70">Published</span><span>{nats.messaging.published}</span>
          <span class="opacity-70">Received</span><span>{nats.messaging.received}</span>
          <span class="opacity-70">Subjects</span><span>{Object.keys(nats.messaging.subjects||{}).length}</span>
          {#each Object.entries(nats.messaging.subjects||{}) as [sub, samples]}
            <span class="col-span-2 truncate text-xs text-slate-400">{sub}: {samples.join(', ')}</span>
          {/each}
        </div>
      {:else}<div class="opacity-70">No NATS data</div>{/if}
    </section>
  <section class="space-y-2">
      <h3 class="font-medium text-amber-300">Pipeline / Autosolve / QUIC</h3>
      <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {#if pipeline.length}
          {#each pipeline as row}
            <span class="opacity-70">{row.stage} avg</span><span>{row.count? (row.sum/row.count).toFixed(1):'0'} ms ({row.count})</span>
          {/each}
        {:else}<span class="col-span-2 opacity-70">No pipeline data</span>{/if}
        {#if autosolve}
          <span class="opacity-70">Auto p50</span><span>{autosolve.p50.toFixed(1)} ms</span>
          <span class="opacity-70">Auto p90</span><span>{autosolve.p90.toFixed(1)} ms</span>
          <span class="opacity-70">Auto p99</span><span>{autosolve.p99.toFixed(1)} ms</span>
        {/if}
        {#if quic}
          <span class="opacity-70">QUIC Conns</span><span>{quic.total_connections}</span>
          <span class="opacity-70">QUIC Streams</span><span>{quic.total_streams}</span>
          <span class="opacity-70">QUIC Avg Lat</span><span>{quic.avg_latency_ms} ms</span>
          <span class="opacity-70">QUIC Errors</span><span>{quic.total_errors}</span>
        {/if}
      </div>
    </section>
    <section class="space-y-2">
      <h3 class="font-medium text-pink-300">Redis Cache</h3>
      {#if redis}
        <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <span class="opacity-70">Status</span><span class={redis.up ? 'text-green-400':'text-red-400'}>{redis.up ? 'up':'down'}</span>
          <span class="opacity-70">Ping</span><span>{redis.last_ping_ms} ms</span>
          {#if redis.last_ok_ts}
            <span class="opacity-70">Last OK</span><span>{new Date(redis.last_ok_ts).toLocaleTimeString()}</span>
          {/if}
          {#if redis.last_error_ts}
            <span class="opacity-70">Last Err</span><span>{new Date(redis.last_error_ts).toLocaleTimeString()}</span>
          {/if}
        </div>
      {:else}<div class="opacity-70 text-xs">No Redis data</div>{/if}
    </section>
  </div>
  <div class="text-xs opacity-60">Last refreshed: {lastRefreshed ? lastRefreshed.toLocaleTimeString() : '—'}</div>
</div>

<style>
  .metrics-widget { font-family: system-ui, sans-serif; }
</style>
