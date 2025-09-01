<!-- Lazy Loading Demo - Shows how to use the lazy loading system -->
<script lang="ts">
  import { 
    LazyLoader, 
    LazyChart, 
    LazyAIAnalysis,
    createLazyStore,
    LAZY_LOAD_PRESETS
  } from '$lib/components/lazy/index.js';
  import type { LazyComponentState } from '$lib/utils/intersection-observer.js';

  // Demonstration data
  const chartData = [
    { label: 'Cases Won', value: 85 },
    { label: 'Cases Lost', value: 12 },
    { label: 'Pending', value: 23 },
    { label: 'Settled', value: 34 }
  ];

  const evidenceData = {
    evidenceId: 'EV-2024-001',
    fileName: 'contract_analysis.pdf',
    fileSize: 2048576,
    uploadDate: '2024-01-15',
    caseId: 'CASE-2024-001'
  };

  // Track lazy states for demonstration
  let chartLazyState: LazyComponentState;
  let analysisLazyState: LazyComponentState;
  let customLazyState: LazyComponentState;

  // Custom lazy store for the manual example
  const customLazyStore = createLazyStore();

  // Mock heavy content for demonstration
  function generateMockContent(type: string) {
    return Array.from({ length: 20 }, (_, i) => ({
      id: i + 1,
      type,
      title: `${type} Item ${i + 1}`,
      description: `This is a mock ${type} item for demonstration purposes.`,
      timestamp: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString()
    }));
  }

  const mockDocuments = generateMockContent('Document');
  const mockEvidence = generateMockContent('Evidence');

  // Stats for demonstration
  let loadedComponents = $derived([
    chartLazyState?.hasBeenVisible,
    analysisLazyState?.hasBeenVisible,
    customLazyState?.hasBeenVisible
  ].filter(Boolean).length);

  let visibleComponents = $derived([
    chartLazyState?.isVisible,
    analysisLazyState?.isVisible,
    customLazyState?.isVisible
  ].filter(Boolean).length);
</script>

<svelte:head>
  <title>Lazy Loading Demo - Legal AI Platform</title>
  <meta name="description" content="Demonstration of lazy loading system for performance optimization" />
</svelte:head>

<div class="demo-container">
  <header class="demo-header">
    <h1>Lazy Loading System Demo</h1>
    <p>Performance optimization through intelligent component loading</p>
    
    <div class="stats-bar">
      <div class="stat-item">
        <span class="stat-value">{loadedComponents}</span>
        <span class="stat-label">Loaded</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">{visibleComponents}</span>
        <span class="stat-label">Visible</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">3</span>
        <span class="stat-label">Total</span>
      </div>
    </div>
  </header>

  <main class="demo-content">
    <!-- Introduction section -->
    <section class="intro-section">
      <h2>How It Works</h2>
      <p>
        The lazy loading system uses the Intersection Observer API to defer rendering 
        of heavy components until they're about to come into view. This improves initial 
        page load times and reduces memory usage.
      </p>
      
      <div class="benefits-grid">
        <div class="benefit-card">
          <div class="benefit-icon">⚡</div>
          <h3>Faster Loading</h3>
          <p>Components only load when needed, reducing initial bundle size.</p>
        </div>
        <div class="benefit-card">
          <div class="benefit-icon">🔋</div>
          <h3>Memory Efficient</h3>
          <p>Lower memory usage by avoiding unnecessary component instantiation.</p>
        </div>
        <div class="benefit-card">
          <div class="benefit-icon">🎯</div>
          <h3>Better UX</h3>
          <p>Users see content faster with progressive loading indicators.</p>
        </div>
        <div class="benefit-card">
          <div class="benefit-icon">📱</div>
          <h3>Mobile Optimized</h3>
          <p>Especially beneficial on slower mobile connections and devices.</p>
        </div>
      </div>
    </section>

    <!-- Scroll down to see lazy loading in action -->
    <div class="scroll-indicator">
      <p>👇 Scroll down to see lazy loading in action</p>
      <div class="scroll-arrow">↓</div>
    </div>

    <!-- Spacer to force scrolling -->
    <div class="spacer">
      <h2>Legal Case Analytics</h2>
      <p>The following chart will load when it comes into view...</p>
    </div>

    <!-- Lazy Chart Example -->
    <section class="demo-section">
      <h2>1. Lazy Chart Component</h2>
      <p>
        Charts can be resource-intensive, especially with large datasets. 
        This chart component loads only when visible.
      </p>
      
      <LazyChart
        {chartData}
        chartType="bar"
        height="400px"
        loadingText="Loading case analytics chart..."
        bind:lazyState={chartLazyState}
        class="demo-chart"
      />

      {#if chartLazyState}
        <div class="component-status">
          <div class="status-item">
            Visible: <span class="status-value">{chartLazyState.isVisible ? '✅' : '❌'}</span>
          </div>
          <div class="status-item">
            Loaded: <span class="status-value">{chartLazyState.hasBeenVisible ? '✅' : '❌'}</span>
          </div>
          <div class="status-item">
            Intersection: <span class="status-value">{Math.round(chartLazyState.intersectionRatio * 100)}%</span>
          </div>
        </div>
      {/if}
    </section>

    <!-- Another spacer -->
    <div class="spacer">
      <h2>AI-Powered Evidence Analysis</h2>
      <p>Complex AI analysis components benefit greatly from lazy loading...</p>
    </div>

    <!-- Lazy AI Analysis Example -->
    <section class="demo-section">
      <h2>2. Lazy AI Analysis Component</h2>
      <p>
        AI analysis components are particularly heavy as they may load models 
        and perform complex computations. Lazy loading prevents unnecessary processing.
      </p>
      
      <LazyAIAnalysis
        analysisType="evidence"
        analysisData={evidenceData}
        model="gemma3-legal"
        temperature={0.7}
        height="600px"
        loadingText="Initializing AI analysis engine..."
        bind:lazyState={analysisLazyState}
        class="demo-analysis"
        onAnalysisComplete={(result) => {
          console.log('Analysis complete:', result);
        }}
      />

      {#if analysisLazyState}
        <div class="component-status">
          <div class="status-item">
            Visible: <span class="status-value">{analysisLazyState.isVisible ? '✅' : '❌'}</span>
          </div>
          <div class="status-item">
            Loaded: <span class="status-value">{analysisLazyState.hasBeenVisible ? '✅' : '❌'}</span>
          </div>
          <div class="status-item">
            Intersection: <span class="status-value">{Math.round(analysisLazyState.intersectionRatio * 100)}%</span>
          </div>
        </div>
      {/if}
    </section>

    <!-- Custom lazy loading example -->
    <div class="spacer">
      <h2>Custom Lazy Loading</h2>
      <p>You can also create custom lazy-loaded content with the base LazyLoader component...</p>
    </div>

    <section class="demo-section">
      <h2>3. Custom Lazy Content</h2>
      <p>
        The base LazyLoader component can wrap any content. Here's a custom 
        implementation showing a document list.
      </p>
      
      <LazyLoader
        preset="NORMAL"
        placeholderHeight="500px"
        loadingText="Loading document repository..."
        bind:lazyState={customLazyState}
        class="demo-custom"
      >
        <div class="document-grid">
          <h3>Document Repository</h3>
          <div class="document-list">
            {#each mockDocuments.slice(0, 8) as doc}
              <div class="document-card">
                <div class="doc-icon">📄</div>
                <div class="doc-info">
                  <h4>{doc.title}</h4>
                  <p>{doc.description}</p>
                  <small>{new Date(doc.timestamp).toLocaleDateString()}</small>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </LazyLoader>

      {#if customLazyState}
        <div class="component-status">
          <div class="status-item">
            Visible: <span class="status-value">{customLazyState.isVisible ? '✅' : '❌'}</span>
          </div>
          <div class="status-item">
            Loaded: <span class="status-value">{customLazyState.hasBeenVisible ? '✅' : '❌'}</span>
          </div>
          <div class="status-item">
            Intersection: <span class="status-value">{Math.round(customLazyState.intersectionRatio * 100)}%</span>
          </div>
        </div>
      {/if}
    </section>

    <!-- Configuration examples -->
    <section class="demo-section">
      <h2>4. Configuration Options</h2>
      <p>The lazy loading system offers various presets and customization options:</p>
      
      <div class="config-grid">
        {#each Object.entries(LAZY_LOAD_PRESETS) as [preset, config]}
          <div class="config-card">
            <h4>{preset}</h4>
            <div class="config-details">
              <div class="config-item">
                <span>Root Margin:</span>
                <code>{config.rootMargin}</code>
              </div>
              <div class="config-item">
                <span>Threshold:</span>
                <code>{Array.isArray(config.threshold) ? config.threshold.join(', ') : config.threshold}</code>
              </div>
              <div class="config-item">
                <span>Once:</span>
                <code>{config.once}</code>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Implementation guide -->
    <section class="demo-section">
      <h2>5. Implementation Guide</h2>
      
      <div class="code-example">
        <h4>Basic Usage</h4>
        <pre><code>{`<script>
  import { LazyLoader } from '$lib/components/lazy';
</script>

<LazyLoader
  preset="NORMAL"
  placeholderHeight="400px"
  loadingText="Loading content..."
>
  <!-- Your heavy content here -->
  <YourHeavyComponent />
</LazyLoader>`}</code></pre>
      </div>

      <div class="code-example">
        <h4>With State Binding</h4>
        <pre><code>{`<script>
  let lazyState;
</script>

<LazyLoader bind:lazyState>
  <YourComponent />
</LazyLoader>

{#if lazyState?.hasBeenVisible}
  <p>Component has loaded!</p>
{/if}`}</code></pre>
      </div>

      <div class="code-example">
        <h4>Custom Options</h4>
        <pre><code>{`<LazyLoader
  customOptions={{
    rootMargin: '100px',
    threshold: 0.5,
    once: true
  }}
  onLoad={() => console.log('Component loaded!')}
>
  <YourComponent />
</LazyLoader>`}</code></pre>
      </div>
    </section>
  </main>
</div>

<style>
  .demo-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  }

  .demo-header {
    text-align: center;
    padding: 48px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    margin-bottom: 48px;
  }

  .demo-header h1 {
    font-size: 48px;
    font-weight: 700;
    margin: 0 0 12px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .demo-header p {
    font-size: 18px;
    color: rgba(0, 0, 0, 0.6);
    margin: 0 0 32px 0;
  }

  .stats-bar {
    display: flex;
    justify-content: center;
    gap: 32px;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .stat-value {
    font-size: 32px;
    font-weight: 700;
    color: #667eea;
  }

  .stat-label {
    font-size: 14px;
    color: rgba(0, 0, 0, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .demo-content {
    display: flex;
    flex-direction: column;
    gap: 96px;
  }

  .intro-section h2 {
    font-size: 32px;
    margin: 0 0 16px 0;
  }

  .intro-section p {
    font-size: 16px;
    line-height: 1.6;
    color: rgba(0, 0, 0, 0.7);
    margin-bottom: 32px;
  }

  .benefits-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 24px;
  }

  .benefit-card {
    padding: 24px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 12px;
    text-align: center;
  }

  .benefit-icon {
    font-size: 48px;
    margin-bottom: 16px;
  }

  .benefit-card h3 {
    font-size: 18px;
    margin: 0 0 12px 0;
  }

  .benefit-card p {
    font-size: 14px;
    color: rgba(0, 0, 0, 0.6);
    margin: 0;
  }

  .scroll-indicator {
    text-align: center;
    padding: 48px 0;
    color: rgba(0, 0, 0, 0.5);
  }

  .scroll-indicator p {
    margin: 0 0 16px 0;
    font-size: 18px;
  }

  .scroll-arrow {
    font-size: 32px;
    animation: bounce 2s infinite;
  }

  .spacer {
    padding: 96px 0;
    text-align: center;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
  }

  .spacer h2 {
    font-size: 28px;
    margin: 0 0 16px 0;
  }

  .spacer p {
    font-size: 16px;
    color: rgba(0, 0, 0, 0.6);
    margin: 0;
  }

  .demo-section {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .demo-section h2 {
    font-size: 24px;
    margin: 0;
  }

  .demo-section p {
    font-size: 16px;
    color: rgba(0, 0, 0, 0.7);
    margin: 0;
  }

  .component-status {
    display: flex;
    gap: 24px;
    padding: 16px;
    background: rgba(0, 0, 0, 0.02);
    border-radius: 8px;
    margin-top: 16px;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
  }

  .status-value {
    font-weight: 600;
    font-family: monospace;
  }

  .document-grid h3 {
    margin: 0 0 24px 0;
    font-size: 20px;
  }

  .document-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }

  .document-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    background: white;
  }

  .doc-icon {
    font-size: 32px;
  }

  .doc-info h4 {
    margin: 0 0 8px 0;
    font-size: 16px;
  }

  .doc-info p {
    margin: 0 0 8px 0;
    font-size: 14px;
    color: rgba(0, 0, 0, 0.6);
  }

  .doc-info small {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.5);
  }

  .config-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
  }

  .config-card {
    padding: 20px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.02);
  }

  .config-card h4 {
    margin: 0 0 16px 0;
    font-size: 16px;
    font-weight: 600;
    color: #667eea;
  }

  .config-details {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .config-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
  }

  .config-item code {
    background: rgba(0, 0, 0, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
  }

  .code-example {
    margin-bottom: 32px;
  }

  .code-example h4 {
    margin: 0 0 12px 0;
    font-size: 16px;
  }

  .code-example pre {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 24px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 0;
  }

  .code-example code {
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.5;
  }

  @keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-10px);
    }
    60% {
      transform: translateY(-5px);
    }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .demo-container {
      padding: 16px;
    }

    .demo-header h1 {
      font-size: 36px;
    }

    .stats-bar {
      gap: 16px;
    }

    .component-status {
      flex-direction: column;
      gap: 12px;
    }

    .benefits-grid {
      grid-template-columns: 1fr;
    }

    .config-grid {
      grid-template-columns: 1fr;
    }
  }
</style>