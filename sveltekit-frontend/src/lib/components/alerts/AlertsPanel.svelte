<script lang="ts">

  interface Props {
    class?: string;
    children?: import('svelte').Snippet;
  }
  import { onMount } from 'svelte';
let alerts = $state<any[] >([]);
let sustained = $state<{ sustainedP99Breaches:number; threshold:number; lastP99OkTs:number } | null >(null);
let loading = $state(true);
let error = $state<string | null >(null);
let autoRefresh = $state(true);
let interval = $state<any;

  async function load(){
    try {
      const res >(await fetch('/api/v1/alerts'));
      const data = await res.json();
      alerts = data.alerts || [];
      const quicRes = await fetch('/api/v1/quic/push', { method:'POST', body: JSON.stringify({ latencySamples: [] }), headers:{'content-type':'application/json'} });
      const quicData = await quicRes.json();
      sustained = quicData.sustainedP99;
      loading = false;
    } catch(e:any){ error = e.message; loading=false; }
  }
  function fmt(ts:number){ return new Date(ts).toLocaleTimeString(); }
  onMount(()=>{ load(); interval = setInterval(()=>{ if(autoRefresh) load(); }, 5000); return ()=> clearInterval(interval); });
</script>

<div class="alerts-panel p-3 border rounded bg-white dark:bg-neutral-900 text-sm space-y-3">
  <div class="flex items-center justify-between">
    <h3 class="font-semibold">Alerts</h3>
    <div class="flex items-center gap-3">
      {#if sustained}
        <span class="px-2 py-1 rounded text-xs font-medium" class:sustained-breach={sustained.sustainedP99Breaches>=sustained.threshold}>
          p99 streak: {sustained.sustainedP99Breaches}/{sustained.threshold}
        </span>
      {/if}
      <button on:onclick={() => autoRefresh = !autoRefresh} class="text-xs border px-2 py-1 rounded hover:bg-neutral-100 dark:hover:bg-neutral-800">{autoRefresh? 'Pause':'Resume'}</button>
      <button on:onclick={load} class="text-xs border px-2 py-1 rounded hover:bg-neutral-100 dark:hover:bg-neutral-800">Refresh</button>
    </div>
  </div>
  {#if loading}
    <div>Loading...</div>
  {:else if error}
    <div class="text-red-600">Error: {error}</div>
  {:else if alerts.length === 0}
    <div class="text-neutral-500">No alerts yet.</div>
  {:else}
    <ul class="space-y-2 max-h-72 overflow-auto">
      {#each alerts as a}
        <li class="border px-2 py-1 rounded flex items-start gap-2" data-severity={a.severity}>
          <span class="text-[10px] mt-0.5 px-1 rounded bg-neutral-200 dark:bg-neutral-700 capitalize">{a.severity}</span>
          <div class="flex-1">
            <div class="font-mono text-xs">{a.type}</div>
            <div class="text-neutral-700 dark:text-neutral-300">{a.message}</div>
            <div class="text-[10px] text-neutral-500">{fmt(a.ts)}</div>
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .alerts-panel [data-severity="critical"] { border-color: #dc2626; }
  .alerts-panel [data-severity="warn"] { border-color: #d97706; }
  .alerts-panel [data-severity="info"] { border-color: #3b82f6; }
  .sustained-breach { background:#dc2626; color:#fff; }
</style>
