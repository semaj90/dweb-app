<script lang="ts">
  import '../app.css';
</script>

<div class="app">
  <header>
    <h1>YoRHa Legal AI</h1>
    <nav class="main-nav">
      <a href="/">Home</a>
      <a href="/yorha-command-center">YoRHa Command Center</a>
      <a href="/demo/enhanced-rag-semantic">Enhanced RAG Demo</a>
      <a href="/endpoints">Endpoints</a>
    </nav>
  </header>
  <main>
    <slot />
  </main>
</div>

<style>
  .app {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
  }
  header {
    background: #1a1a1a;
    color: #ffd700;
    padding: 1rem;
  }
  .main-nav {
    margin-top: 1rem;
    display: flex;
    gap: 1rem;
  }
  .main-nav a {
    color: #ffd700;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border: 1px solid #ffd700;
    border-radius: 4px;
    transition: all 0.2s ease;
  }
  .main-nav a:hover {
    background: #ffd700;
    color: #1a1a1a;
  }
  main {
    padding: 1rem;
  }
</style>