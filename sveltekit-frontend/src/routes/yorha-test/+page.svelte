<!-- Minimal YoRHa Test Page -->
<script lang="ts">
  import { onMount } from 'svelte';
  
  let message = $state('YoRHa Test Page Loading...');
  
  onMount(() => {
    message = 'YoRHa Test Page Loaded Successfully!';
  });
</script>

<svelte:head>
  <title>YoRHa Test</title>
</svelte:head>

<div class="yorha-test">
  <h1>YoRHa Test Page</h1>
  <p>{message}</p>
  <div class="test-info">
    <h2>Component Test Status:</h2>
    <ul>
      <li>Base Route: ✅ Working</li>
      <li>Svelte 5 Syntax: ✅ Working</li>
      <li>Page Loading: ✅ Working</li>
    </ul>
  </div>
</div>

<style>
  .yorha-test {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    padding: 40px;
  }
  
  h1 {
    color: #ffd700;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 2px solid #ffd700;
    padding-bottom: 10px;
  }
  
  .test-info {
    background: #1a1a1a;
    border: 2px solid #ffd700;
    padding: 20px;
    margin-top: 20px;
  }
  
  h2 {
    color: #00ff41;
    margin-top: 0;
  }
  
  li {
    margin: 8px 0;
    color: #e0e0e0;
  }
</style>