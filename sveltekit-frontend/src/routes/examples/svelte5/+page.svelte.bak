<script lang="ts">
  import Svelte5Examples from '$lib/components/examples/Svelte5Examples.svelte';
  import { useMediaQuery } from '$lib/utils/media-query.svelte';
  
  // Example of custom media query
  const isWideScreen = useMediaQuery('(min-width: 1400px)');
</script>

<svelte:head>
  <title>Svelte 5 Examples - Children & Media Queries</title>
</svelte:head>

<!-- Using the component with snippets -->
<Svelte5Examples title="Svelte 5 Demo">
  
  <!-- Custom header snippet with parameters -->
  {#snippet header({ isMobile })}
    <div class="custom-header">
      <h1>🚀 Svelte 5 Features</h1>
      <div class="device-info">
        {#if isMobile}
          📱 Mobile View
        {:else}
          🖥️ Desktop View
        {/if}
        {#if isWideScreen.matches}
          | Ultra-wide!
        {/if}
      </div>
    </div>
  {/snippet}
  
  <!-- Sidebar snippet -->
  {#snippet sidebar()}
    <nav>
      <h3>Navigation</h3>
      <ul>
        <li><a href="#children">Children Pattern</a></li>
        <li><a href="#media">Media Queries</a></li>
        <li><a href="#runes">Svelte Runes</a></li>
      </ul>
    </nav>
  {/snippet}
  
  <!-- Main content snippet -->
  {#snippet children()}
    <div class="examples-content">
      
      <section id="children">
        <h2>🧩 Children Pattern</h2>
        <p>Svelte 5 replaces <code>&lt;slot&gt;</code> with <code>{`{#snippet}`}</code>:</p>
        <pre><code>{`<!-- ❌ Svelte 4 -->
<slot />

<!-- ✅ Svelte 5 -->
{#if children}
  {@render children()}
{/if}`}</code></pre>
      </section>
      
      <section id="media">
        <h2>📱 Media Query Reactivity</h2>
        <p>Reactive media queries using Svelte 5 runes:</p>
        <pre><code>{`const mobile = useIsMobile();
const desktop = useIsDesktop();

// Reactive derived values
let layout = $derived(mobile.matches ? 'mobile' : 'desktop');`}</code></pre>
        
        <div class="live-example">
          <h4>Live Example:</h4>
          <p>Current screen: <strong>{isWideScreen.matches ? 'Ultra-wide' : 'Standard'}</strong></p>
          <p>Screen width query: <code>(min-width: 1400px)</code></p>
        </div>
      </section>
      
      <section id="runes">
        <h2>✨ Svelte 5 Runes</h2>
        <ul>
          <li><code>$state()</code> - Reactive state</li>
          <li><code>$derived()</code> - Computed values</li>
          <li><code>$effect()</code> - Side effects</li>
          <li><code>$props()</code> - Component props</li>
        </ul>
      </section>
      
    </div>
  {/snippet}
  
  <!-- Footer snippet -->
  {#snippet footer()}
    <p>Built with Svelte 5 • Reactive Media Queries • Modern Patterns</p>
  {/snippet}
  
</Svelte5Examples>

<style>
  .custom-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
  }
  
  .device-info {
    font-family: 'Roboto Mono', monospace;
    font-size: 0.875rem;
    color: #666;
  }
  
  .examples-content {
    max-width: 800px;
  }
  
  .examples-content section {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #eee;
  }
  
  .examples-content h2 {
    color: #2563eb;
    margin-bottom: 1rem;
  }
  
  .examples-content code {
    background: #f1f5f9;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.875rem;
  }
  
  .examples-content pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    margin: 1rem 0;
  }
  
  .examples-content pre code {
    background: none;
    color: inherit;
    padding: 0;
  }
  
  .live-example {
    background: #f0f9ff;
    border: 1px solid #0ea5e9;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
  }
  
  .live-example h4 {
    margin: 0 0 0.5rem 0;
    color: #0c4a6e;
  }
  
  nav ul {
    list-style: none;
    padding: 0;
  }
  
  nav li {
    margin: 0.5rem 0;
  }
  
  nav a {
    color: #3b82f6;
    text-decoration: none;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    transition: background-color 0.2s;
  }
  
  nav a:hover {
    background: #dbeafe;
  }
</style>