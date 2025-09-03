<script lang="ts"></script>
  export let data: { message: string; path: string; targetExample: string; now: string };
</script>

<div class="p-6 space-y-4 border border-neutral-700 rounded-lg bg-neutral-900/50">
  <h1 class="text-xl font-bold">Proxy Diagnostic</h1>
  <p>{data.message}</p>
  <p class="text-sm text-neutral-400">Path: {data.path}</p>
  <p class="text-sm text-neutral-400">Example target: {data.targetExample}</p>
  <p class="text-xs text-neutral-500">Server time: {data.now}</p>
</div>
