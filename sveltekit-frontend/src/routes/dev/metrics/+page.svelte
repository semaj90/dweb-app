<script lang="ts">
  import MetricsDashboardWidget from '$lib/components/metrics/MetricsDashboardWidget.svelte';
</script>

<svelte:head>
  <title>Dev Metrics • Legal AI</title>
</svelte:head>

<div class="p-6 max-w-5xl mx-auto space-y-6">
  <h1 class="text-2xl font-bold mb-2">Development Metrics</h1>
  <p class="text-sm opacity-70">Live AI & messaging subsystem performance snapshot (auto-refresh 10s).</p>
  <MetricsDashboardWidget />
</div>

<style>
  :global(body){ background: #0f1115; }
</style>
