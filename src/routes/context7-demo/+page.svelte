<script lang="ts">
  import { onMount } from 'svelte';

  // Context7 integration state
  let connectionStatus = $state('DISCONNECTED');
  let isConnecting = $state(false);
  let demoFeatures = $state([
    {
      id: 'search',
      name: 'Semantic Search',
      description: 'Search across indexed documents using natural language',
      status: 'ready',
      lastUsed: null
    },
    {
      id: 'index',
      name: 'Document Indexing',
      description: 'Index new documents for semantic search',
      status: 'ready',
      lastUsed: null
    },
    {
      id: 'memory',
      name: 'Memory Management',
      description: 'Persistent context and conversation memory',
      status: 'ready',
      lastUsed: null
    },
    {
      id: 'vector',
      name: 'Vector Operations',
      description: 'Vector similarity and embedding operations',
      status: 'ready',
      lastUsed: null
    }
  ]);

  let searchQuery = $state('');
  let searchResults = $state([]);
  let isSearching = $state(false);
  let selectedFeature = $state('search');
  let demoOutput = $state('Context7 MCP Integration Demo initialized...\n');

  async function connectToContext7() {
    isConnecting = true;
    connectionStatus = 'CONNECTING';
    demoOutput += 'Attempting Context7 MCP connection...\n';

    try {
      // Simulate connection attempt
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const success = Math.random() > 0.2; // 80% success rate
      
      if (success) {
        connectionStatus = 'CONNECTED';
        demoOutput += 'Connection successful!\nContext7 MCP server responding\nAll features available\n';
        
        // Enable all features
        demoFeatures = demoFeatures.map(feature => ({
          ...feature,
          status: 'ready'
        }));
      } else {
        connectionStatus = 'FAILED';
        demoOutput += 'Connection failed!\nCheck Context7 server status\nRetry connection\n';
      }
    } catch (error) {
      connectionStatus = 'ERROR';
      demoOutput += `Connection error: ${error.message}\n`;
    }

    isConnecting = false;
  }

  async function testFeature(featureId: string) {
    const feature = demoFeatures.find(f => f.id === featureId);
    if (!feature || connectionStatus !== 'CONNECTED') return;

    // Update feature status
    feature.status = 'testing';
    feature.lastUsed = new Date().toLocaleTimeString();
    demoFeatures = [...demoFeatures];
    
    demoOutput += `Testing ${feature.name}...\n`;

    try {
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const success = Math.random() > 0.15; // 85% success rate
      
      if (success) {
        feature.status = 'success';
        demoOutput += `${feature.name} test completed successfully!\n`;
        
        // Simulate feature-specific results
        switch (featureId) {
          case 'search':
            demoOutput += 'Found 12 relevant documents\nSemantic similarity scores calculated\n';
            break;
          case 'index':
            demoOutput += 'Document indexed successfully\nVector embeddings generated\n';
            break;
          case 'memory':
            demoOutput += 'Memory context updated\nConversation history preserved\n';
            break;
          case 'vector':
            demoOutput += 'Vector operations completed\nSimilarity calculations processed\n';
            break;
        }
      } else {
        feature.status = 'error';
        demoOutput += `${feature.name} test failed!\nRetry recommended\n`;
      }
    } catch (error) {
      feature.status = 'error';
      demoOutput += `${feature.name} error: ${error.message}\n`;
    }

    demoFeatures = [...demoFeatures];
  }

  async function performSearch() {
    if (!searchQuery.trim() || connectionStatus !== 'CONNECTED') return;

    isSearching = true;
    demoOutput += `Performing semantic search: "${searchQuery}"\n`;

    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Simulate search results
      const mockResults = [
        {
          id: 1,
          title: 'Legal Contract Analysis Framework',
          excerpt: 'Comprehensive framework for analyzing legal contracts using AI...',
          similarity: 0.92,
          source: 'legal-docs/contracts/framework.pdf'
        },
        {
          id: 2,
          title: 'Semantic Search in Legal Documents',
          excerpt: 'Advanced techniques for semantic search across legal document collections...',
          similarity: 0.87,
          source: 'legal-docs/research/semantic-search.pdf'
        },
        {
          id: 3,
          title: 'AI-Powered Legal Research Tools',
          excerpt: 'Tools and methodologies for AI-powered legal research and analysis...',
          similarity: 0.81,
          source: 'legal-docs/tools/ai-research.pdf'
        }
      ];

      searchResults = mockResults;
      demoOutput += `Search completed! Found ${mockResults.length} results\n`;
      demoOutput += `Top similarity: ${(mockResults[0].similarity * 100).toFixed(1)}%\n`;
      
      // Update search feature
      const searchFeature = demoFeatures.find(f => f.id === 'search');
      if (searchFeature) {
        searchFeature.status = 'success';
        searchFeature.lastUsed = new Date().toLocaleTimeString();
        demoFeatures = [...demoFeatures];
      }
      
    } catch (error) {
      demoOutput += `Search error: ${error.message}\n`;
    }

    isSearching = false;
  }

  function clearOutput() {
    demoOutput = 'Output cleared...\n';
    searchResults = [];
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'ready': return '#4ecdc4';
      case 'testing': return '#ffbf00';
      case 'success': return '#00ff41';
      case 'error': return '#ff6b6b';
      default: return '#888';
    }
  }

  function getStatusIcon(status: string): string {
    switch (status) {
      case 'ready': return '⚡';
      case 'testing': return '🔄';
      case 'success': return '✅';
      case 'error': return '❌';
      default: return '❓';
    }
  }
</script>

<svelte:head>
  <title>Context7 MCP Integration Demo</title>
</svelte:head>

<div class="context7-demo">
  <div class="demo-header">
    <h1 class="page-title">
      <span class="title-icon">🔗</span>
      CONTEXT7 MCP INTEGRATION
    </h1>
    <div class="connection-status">
      <span class="status-label">CONNECTION:</span>
      <span class="status-value {connectionStatus.toLowerCase()}">{connectionStatus}</span>
    </div>
  </div>

  <div class="demo-grid">
    <!-- Connection Control -->
    <section class="connection-panel">
      <h2 class="section-title">MCP CONNECTION</h2>
      
      <div class="connection-info">
        <div class="info-item">
          <span class="info-label">PROTOCOL:</span>
          <span class="info-value">Model Context Protocol</span>
        </div>
        <div class="info-item">
          <span class="info-label">SERVER:</span>
          <span class="info-value">Context7 MCP Server</span>
        </div>
        <div class="info-item">
          <span class="info-label">VERSION:</span>
          <span class="info-value">1.0.0</span>
        </div>
      </div>

      <button
        class="connect-button {connectionStatus.toLowerCase()}"
        onclick={connectToContext7}
        disabled={isConnecting}
      >
        <div class="button-icon">
          {isConnecting ? '⏳' : connectionStatus === 'CONNECTED' ? '🔗' : '🔌'}
        </div>
        <div class="button-text">
          {isConnecting ? 'CONNECTING...' : connectionStatus === 'CONNECTED' ? 'RECONNECT' : 'CONNECT TO CONTEXT7'}
        </div>
      </button>
    </section>

    <!-- Feature Testing -->
    <section class="features-panel">
      <h2 class="section-title">MCP FEATURES</h2>
      
      <div class="features-grid">
        {#each demoFeatures as feature}
          <div class="feature-card {feature.status}">
            <div class="feature-header">
              <div class="feature-name">{feature.name}</div>
              <div class="feature-status" style="color: {getStatusColor(feature.status)}">
                {getStatusIcon(feature.status)} {feature.status.toUpperCase()}
              </div>
            </div>
            
            <div class="feature-description">{feature.description}</div>
            
            <div class="feature-footer">
              {#if feature.lastUsed}
                <div class="feature-last-used">Last used: {feature.lastUsed}</div>
              {/if}
              <button
                class="feature-test-button"
                onclick={() => testFeature(feature.id)}
                disabled={connectionStatus !== 'CONNECTED' || feature.status === 'testing'}
              >
                {feature.status === 'testing' ? 'TESTING...' : 'TEST FEATURE'}
              </button>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Search Demo -->
    <section class="search-panel">
      <h2 class="section-title">SEMANTIC SEARCH DEMO</h2>
      
      <div class="search-container">
        <div class="search-input-group">
          <input
            bind:value={searchQuery}
            placeholder="Enter semantic search query..."
            class="search-input"
            onkeydown={(e) => e.key === 'Enter' && performSearch()}
            disabled={connectionStatus !== 'CONNECTED'}
          />
          <button
            class="search-button"
            onclick={performSearch}
            disabled={connectionStatus !== 'CONNECTED' || isSearching || !searchQuery.trim()}
          >
            <div class="button-icon">{isSearching ? '🔄' : '🔍'}</div>
            <div class="button-text">{isSearching ? 'SEARCHING...' : 'SEARCH'}</div>
          </button>
        </div>

        <div class="search-results">
          {#if searchResults.length > 0}
            <div class="results-header">
              <h3>Search Results ({searchResults.length})</h3>
            </div>
            {#each searchResults as result}
              <div class="result-item">
                <div class="result-header">
                  <div class="result-title">{result.title}</div>
                  <div class="result-similarity">{(result.similarity * 100).toFixed(1)}%</div>
                </div>
                <div class="result-excerpt">{result.excerpt}</div>
                <div class="result-source">Source: {result.source}</div>
              </div>
            {/each}
          {:else if searchQuery && !isSearching}
            <div class="no-results">
              <div class="no-results-icon">🔍</div>
              <div class="no-results-text">No results found</div>
              <div class="no-results-subtext">Try connecting to Context7 first</div>
            </div>
          {/if}
        </div>
      </div>
    </section>

    <!-- Output Console -->
    <section class="console-panel">
      <div class="console-header">
        <h2 class="section-title">SYSTEM OUTPUT</h2>
        <button class="clear-button" onclick={clearOutput}>
          🗑️ CLEAR
        </button>
      </div>
      
      <div class="console">
        <pre class="console-output">{demoOutput}</pre>
      </div>
    </section>
  </div>

  <!-- Back Navigation -->
  <div class="navigation-footer">
    <a href="/" class="back-button">
      <span class="button-icon">⬅️</span>
      RETURN TO COMMAND CENTER
    </a>
  </div>
</div>

<style>
  .context7-demo {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    padding: 2rem;
  }

  .demo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3rem;
    padding: 2rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 2px solid #4ecdc4;
    border-radius: 8px;
  }

  .page-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    background: linear-gradient(45deg, #4ecdc4, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px #4ecdc4);
  }

  .connection-status {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
  }

  .status-label {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  .status-value {
    font-size: 1.2rem;
    font-weight: 700;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    letter-spacing: 1px;
  }

  .status-value.disconnected {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }

  .status-value.connecting {
    background: rgba(255, 191, 0, 0.2);
    color: #ffbf00;
    border: 1px solid #ffbf00;
    animation: pulse 1s infinite;
  }

  .status-value.connected {
    background: rgba(0, 255, 65, 0.2);
    color: #00ff41;
    border: 1px solid #00ff41;
  }

  .status-value.failed,
  .status-value.error {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }

  .demo-grid {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4ecdc4;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #4ecdc4;
  }

  /* Connection Panel */
  .connection-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .connection-info {
    margin-bottom: 2rem;
  }

  .info-item {
    display: flex;
    justify-content: space-between;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .info-label {
    color: #f0f0f0;
    opacity: 0.8;
  }

  .info-value {
    color: #4ecdc4;
    font-weight: 600;
  }

  .connect-button {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(145deg, #4ecdc4, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 2rem;
    border-radius: 6px;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .connect-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
  }

  .connect-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .connect-button.connected {
    background: linear-gradient(145deg, #00ff41, #4ecdc4);
  }

  .connect-button.connecting {
    animation: pulse 1.5s ease-in-out infinite;
  }

  /* Features Panel */
  .features-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .feature-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
  }

  .feature-card:hover {
    background: rgba(78, 205, 196, 0.05);
    border-color: rgba(78, 205, 196, 0.3);
  }

  .feature-card.testing {
    border-color: #ffbf00;
    animation: pulse 2s ease-in-out infinite;
  }

  .feature-card.success {
    border-color: #00ff41;
  }

  .feature-card.error {
    border-color: #ff6b6b;
  }

  .feature-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .feature-name {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
  }

  .feature-status {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .feature-description {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
    margin-bottom: 1rem;
    line-height: 1.4;
  }

  .feature-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .feature-last-used {
    font-size: 0.7rem;
    color: #f0f0f0;
    opacity: 0.6;
  }

  .feature-test-button {
    background: transparent;
    border: 2px solid #4ecdc4;
    color: #4ecdc4;
    padding: 0.5rem 1rem;
    font-family: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .feature-test-button:hover:not(:disabled) {
    background: #4ecdc4;
    color: #0a0a0a;
  }

  .feature-test-button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Search Panel */
  .search-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .search-container {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .search-input-group {
    display: flex;
    gap: 1rem;
  }

  .search-input {
    flex: 1;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 1rem;
    border-radius: 4px;
    transition: all 0.3s ease;
  }

  .search-input:focus {
    outline: none;
    border-color: #4ecdc4;
    box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
  }

  .search-input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .search-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(145deg, #4ecdc4, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .search-button:hover:not(:disabled) {
    transform: scale(1.05);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
  }

  .search-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .search-results {
    min-height: 200px;
  }

  .results-header h3 {
    font-size: 1.1rem;
    color: #4ecdc4;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .result-item {
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .result-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
  }

  .result-similarity {
    background: rgba(0, 255, 65, 0.2);
    color: #00ff41;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid #00ff41;
  }

  .result-excerpt {
    font-size: 0.95rem;
    color: #f0f0f0;
    opacity: 0.9;
    margin-bottom: 0.8rem;
    line-height: 1.4;
  }

  .result-source {
    font-size: 0.8rem;
    color: #4ecdc4;
    opacity: 0.8;
    font-family: 'Courier New', monospace;
  }

  .no-results {
    text-align: center;
    padding: 3rem;
    color: #f0f0f0;
    opacity: 0.6;
  }

  .no-results-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .no-results-text {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .no-results-subtext {
    font-size: 0.9rem;
  }

  /* Console Panel */
  .console-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .console-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .clear-button {
    background: transparent;
    border: 2px solid #ff6b6b;
    color: #ff6b6b;
    padding: 0.5rem 1rem;
    font-family: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .clear-button:hover {
    background: #ff6b6b;
    color: #0a0a0a;
  }

  .console {
    background: #000000;
    border: 1px solid #4ecdc4;
    border-radius: 4px;
    padding: 1rem;
    height: 200px;
    overflow-y: auto;
  }

  .console-output {
    color: #00ff41;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    line-height: 1.4;
    white-space: pre-wrap;
    margin: 0;
  }

  /* Navigation Footer */
  .navigation-footer {
    text-align: center;
    margin-top: 3rem;
  }

  .back-button {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(145deg, #4ecdc4, #00ff41);
    color: #0a0a0a;
    padding: 1rem 2rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .demo-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .page-title {
      font-size: 2rem;
    }

    .demo-grid {
      grid-template-columns: 1fr;
    }

    .features-grid {
      grid-template-columns: 1fr;
    }

    .search-input-group {
      flex-direction: column;
    }
  }
</style>