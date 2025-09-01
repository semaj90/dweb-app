<script lang="ts">
  let { data = $bindable() } = $props(); // { endpoints: { name: string; path: string; healthy: boolean; message?: string }[] };
</script>

<div class="endpoints-page">
  <h1 class="page-title">Endpoints & Status</h1>

  <div class="endpoint-list">
    {#each data.endpoints as ep}
      <div class="endpoint-card {ep.healthy ? 'ok' : 'fail'}">
        <h2>{ep.name}</h2>
        <p class="path">{ep.path}</p>
        <p class="status">
          {ep.healthy ? '✅ Healthy' : '❌ Down'} {ep.message ? `(${ep.message})` : ''}
        </p>
      </div>
    {/each}
  </div>
</div>

<style>
  .endpoints-page {
    padding: 2rem;
  }
  .page-title {
    font-size: 1.8rem;
    margin-bottom: 1rem;
    color: #ffd700;
  }
  .endpoint-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
  }
  .endpoint-card {
    border-radius: var(--radius-lg, 8px);
    padding: 1rem;
    box-shadow: var(--shadow-md, 0 4px 6px rgba(0, 0, 0, 0.3));
    background: var(--surface, #2a2a2a);
    color: var(--text-primary, #e0e0e0);
    border: 1px solid #444;
  }
  .endpoint-card.ok { 
    border-left: 6px solid var(--success, #00ff41); 
  }
  .endpoint-card.fail { 
    border-left: 6px solid var(--danger, #ff0041); 
  }
  .endpoint-card h2 {
    margin: 0 0 0.5rem 0;
    font-size: 1.2rem;
    color: #ffd700;
  }
  .path {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--muted, #b0b0b0);
    margin: 0.5rem 0;
    word-break: break-all;
  }
  .status {
    margin-top: 0.5rem;
    font-weight: 600;
    color: var(--text-primary, #e0e0e0);
  }
</style>