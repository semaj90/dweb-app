<script lang="ts">
  import { onMount } from 'svelte';
  type NLPQuantiles = { p50:number; p90:number; p99:number };
  interface NLPStats { embeddings_total:number; cache_hits:number; cache_misses:number; latency: NLPQuantiles; similarity_queries_total:number; hit_ratio:number }
  interface NATSMetricSnapshot { connection:{ status:string; since:number|null; reconnectAttempts:number }; messaging:{ published:number; received:number; subjects: Record<string,string[]> } }

  let loading = true;
  let error: string | null = null;
  let nlp: NLPStats | null = null;
  let nats: NATSMetricSnapshot | null = null;
  let lastRefreshed: Date | null = null;

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

  async function refresh(){
    loading = true; error = null;
    try {
      const [nlpRes, natsRes] = await Promise.allSettled([fetchNLP(), fetchNATS()]);
      if(nlpRes.status==='fulfilled') nlp = nlpRes.value; else error = (nlpRes.reason?.message)||error;
      if(natsRes.status==='fulfilled') nats = natsRes.value; else error = (natsRes.reason?.message)||error;
      lastRefreshed = new Date();
    } catch (e:any){ error = e.message; }
    loading = false;
  }

  let interval: any;
  onMount(()=>{ refresh(); interval = setInterval(refresh, 10000); return ()=> clearInterval(interval); });
</script>

<div class="metrics-widget border rounded-lg p-4 bg-slate-900 text-slate-100 text-sm space-y-4">
  <div class="flex items-center justify-between">
    <h2 class="text-base font-semibold">AI Metrics Dashboard</h2>
    <button class="px-2 py-1 text-xs rounded bg-slate-700 hover:bg-slate-600" on:click={refresh} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>
  </div>
  {#if error}
    <div class="text-red-400">{error}</div>
  {/if}
  <div class="grid md:grid-cols-2 gap-4">
    <section class="space-y-2">
      <h3 class="font-medium text-indigo-300">NLP Embeddings</h3>
      {#if nlp}
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
          <span class="opacity-70">Total</span><span>{nlp.embeddings_total}</span>
          <span class="opacity-70">Cache Hits</span><span>{nlp.cache_hits}</span>
          <span class="opacity-70">Cache Misses</span><span>{nlp.cache_misses}</span>
          <span class="opacity-70">Hit Ratio</span><span>{(nlp.hit_ratio*100).toFixed(1)}%</span>
          <span class="opacity-70">Similarity Queries</span><span>{nlp.similarity_queries_total}</span>
          <span class="opacity-70">Latency p50</span><span>{nlp.latency.p50} ms</span>
          <span class="opacity-70">Latency p90</span><span>{nlp.latency.p90} ms</span>
          <span class="opacity-70">Latency p99</span><span>{nlp.latency.p99} ms</span>
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
  </div>
  <div class="text-xs opacity-60">Last refreshed: {lastRefreshed ? lastRefreshed.toLocaleTimeString() : '—'}</div>
</div>

<style>
  .metrics-widget { font-family: system-ui, sans-serif; }
</style>
