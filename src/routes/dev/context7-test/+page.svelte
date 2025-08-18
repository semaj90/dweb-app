<script lang="ts">
  import { onMount } from 'svelte';

  // Context7 test state
  let isConnected = $state(false);
  let isLoading = $state(false);
  let testResults = $state([]);
  let connectionStatus = $state('DISCONNECTED');
  let testOutput = $state('');
  let selectedTest = $state('ping');

  const testSuites = {
    ping: {
      name: 'Connection Test',
      description: 'Test basic Context7 server connectivity',
      endpoint: '/api/context7/ping'
    },
    search: {
      name: 'Search Test',
      description: 'Test Context7 document search functionality',
      endpoint: '/api/context7/search'
    },
    index: {
      name: 'Index Test', 
      description: 'Test Context7 document indexing',
      endpoint: '/api/context7/index'
    },
    embedding: {
      name: 'Embedding Test',
      description: 'Test Context7 embedding generation',
      endpoint: '/api/context7/embeddings'
    },
    memory: {
      name: 'Memory Test',
      description: 'Test Context7 memory management',
      endpoint: '/api/context7/memory'
    }
  };

  async function testConnection() {
    isLoading = true;
    connectionStatus = 'CONNECTING...';
    testOutput = 'Attempting Context7 connection...\n';
    
    try {
      // Simulate Context7 connection test
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const success = Math.random() > 0.3; // 70% success rate for demo
      
      if (success) {
        isConnected = true;
        connectionStatus = 'CONNECTED';
        testOutput += 'Connection successful!\nContext7 server responding\nMCP integration active\n';
        
        // Add to results
        testResults = [{
          id: Date.now(),
          test: 'Connection',
          status: 'PASSED',
          time: new Date().toLocaleTimeString(),
          message: 'Successfully connected to Context7 server'
        }, ...testResults];
      } else {
        isConnected = false;
        connectionStatus = 'CONNECTION FAILED';
        testOutput += 'Connection failed!\nContext7 server not responding\nCheck server status\n';
        
        testResults = [{
          id: Date.now(),
          test: 'Connection',
          status: 'FAILED',
          time: new Date().toLocaleTimeString(),
          message: 'Failed to connect to Context7 server'
        }, ...testResults];
      }
    } catch (error) {
      isConnected = false;
      connectionStatus = 'ERROR';
      testOutput += `Error: ${error.message}\n`;
    }
    
    isLoading = false;
  }

  async function runTest(testType: string) {
    if (!isConnected) {
      testOutput += 'Error: Not connected to Context7 server\n';
      return;
    }

    const test = testSuites[testType];
    isLoading = true;
    testOutput += `Running ${test.name}...\n`;

    try {
      // Simulate test execution
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const success = Math.random() > 0.2; // 80% success rate
      const responseTime = Math.floor(Math.random() * 500) + 50;
      
      if (success) {
        testOutput += `${test.name} completed successfully (${responseTime}ms)\n`;
        
        testResults = [{
          id: Date.now(),
          test: test.name,
          status: 'PASSED',
          time: new Date().toLocaleTimeString(),
          message: `Test completed in ${responseTime}ms`,
          responseTime
        }, ...testResults];
      } else {
        testOutput += `${test.name} failed - unexpected error\n`;
        
        testResults = [{
          id: Date.now(),
          test: test.name,
          status: 'FAILED',
          time: new Date().toLocaleTimeString(),
          message: 'Test failed with unexpected error'
        }, ...testResults];
      }
    } catch (error) {
      testOutput += `Error running ${test.name}: ${error.message}\n`;
    }
    
    isLoading = false;
  }

  function clearResults() {
    testResults = [];
    testOutput = '';
  }

  function exportResults() {
    const data = {
      timestamp: new Date().toISOString(),
      connectionStatus,
      testResults,
      output: testOutput
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `context7-test-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  onMount(() => {
    testOutput = 'Context7 Test Suite initialized\nReady to run tests...\n';
  });
</script>

<svelte:head>
  <title>Context7 Test Suite</title>
</svelte:head>

<div class="context7-test">
  <div class="test-header">
    <h1 class="page-title">
      <span class="title-icon">🧪</span>
      CONTEXT7 TEST SUITE
    </h1>
    <div class="connection-status">
      <span class="status-label">STATUS:</span>
      <span class="status-value {connectionStatus.toLowerCase().replace(/[^a-z]/g, '')}">{connectionStatus}</span>
    </div>
  </div>

  <div class="test-grid">
    <!-- Connection Panel -->
    <section class="connection-panel">
      <h2 class="section-title">CONNECTION CONTROL</h2>
      <div class="connection-controls">
        <button 
          class="connect-button {isConnected ? 'connected' : ''}"
          onclick={testConnection}
          disabled={isLoading}
        >
          <div class="button-icon">{isConnected ? '🔗' : '🔌'}</div>
          <div class="button-text">
            {isLoading ? 'CONNECTING...' : isConnected ? 'RECONNECT' : 'CONNECT TO CONTEXT7'}
          </div>
        </button>
        
        <div class="connection-info">
          <div class="info-item">
            <span class="info-label">SERVER:</span>
            <span class="info-value">Context7 MCP Server</span>
          </div>
          <div class="info-item">
            <span class="info-label">PROTOCOL:</span>
            <span class="info-value">MCP over WebSocket</span>
          </div>
          <div class="info-item">
            <span class="info-label">VERSION:</span>
            <span class="info-value">1.0.0</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Test Suite Panel -->
    <section class="test-suite-panel">
      <h2 class="section-title">TEST SUITE</h2>
      <div class="test-controls">
        <select bind:value={selectedTest} class="test-selector">
          {#each Object.entries(testSuites) as [key, test]}
            <option value={key}>{test.name}</option>
          {/each}
        </select>
        
        <button 
          class="run-test-button"
          onclick={() => runTest(selectedTest)}
          disabled={!isConnected || isLoading}
        >
          <div class="button-icon">🚀</div>
          <div class="button-text">RUN TEST</div>
        </button>
      </div>

      <div class="test-suite-grid">
        {#each Object.entries(testSuites) as [key, test]}
          <div class="test-item">
            <div class="test-info">
              <div class="test-name">{test.name}</div>
              <div class="test-description">{test.description}</div>
              <div class="test-endpoint">{test.endpoint}</div>
            </div>
            <button 
              class="quick-test-button"
              onclick={() => runTest(key)}
              disabled={!isConnected || isLoading}
            >
              TEST
            </button>
          </div>
        {/each}
      </div>
    </section>

    <!-- Results Panel -->
    <section class="results-panel">
      <div class="results-header">
        <h2 class="section-title">TEST RESULTS</h2>
        <div class="results-controls">
          <button class="control-button" onclick={clearResults}>
            🗑️ CLEAR
          </button>
          <button class="control-button" onclick={exportResults}>
            📥 EXPORT
          </button>
        </div>
      </div>

      <div class="results-stats">
        <div class="stat-item">
          <div class="stat-value">{testResults.length}</div>
          <div class="stat-label">TOTAL TESTS</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{testResults.filter(r => r.status === 'PASSED').length}</div>
          <div class="stat-label">PASSED</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{testResults.filter(r => r.status === 'FAILED').length}</div>
          <div class="stat-label">FAILED</div>
        </div>
      </div>

      <div class="results-list">
        {#each testResults as result}
          <div class="result-item {result.status.toLowerCase()}">
            <div class="result-status">{result.status === 'PASSED' ? '✅' : '❌'}</div>
            <div class="result-content">
              <div class="result-test">{result.test}</div>
              <div class="result-message">{result.message}</div>
              <div class="result-meta">
                <span class="result-time">{result.time}</span>
                {#if result.responseTime}
                  <span class="result-response-time">{result.responseTime}ms</span>
                {/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
    </section>

    <!-- Output Console -->
    <section class="console-panel">
      <h2 class="section-title">CONSOLE OUTPUT</h2>
      <div class="console">
        <pre class="console-output">{testOutput}</pre>
      </div>
    </section>
  </div>

  <!-- Back Navigation -->
  <div class="navigation-footer">
    <a href="/dev/mcp-tools" class="back-button">
      <span class="button-icon">⬅️</span>
      BACK TO MCP TOOLS
    </a>
    <a href="/" class="home-button">
      <span class="button-icon">🏠</span>
      COMMAND CENTER
    </a>
  </div>
</div>

<style>
  .context7-test {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    padding: 2rem;
  }

  .test-header {
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

  .status-value.connectionfailed,
  .status-value.error {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }

  .test-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
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
    margin-bottom: 2rem;
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

  .button-icon {
    font-size: 1.5rem;
  }

  .connection-info {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .info-item {
    display: flex;
    justify-content: space-between;
    padding: 0.8rem;
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

  /* Test Suite Panel */
  .test-suite-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .test-controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .test-selector {
    flex: 1;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    border-radius: 4px;
  }

  .run-test-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #00ff41;
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

  .run-test-button:hover:not(:disabled) {
    background: #4ecdc4;
    transform: scale(1.05);
  }

  .run-test-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .test-suite-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .test-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .test-info {
    flex: 1;
  }

  .test-name {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.5rem;
  }

  .test-description {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
    margin-bottom: 0.5rem;
  }

  .test-endpoint {
    font-size: 0.8rem;
    font-family: 'Courier New', monospace;
    color: #4ecdc4;
    opacity: 0.7;
  }

  .quick-test-button {
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

  .quick-test-button:hover:not(:disabled) {
    background: #4ecdc4;
    color: #0a0a0a;
  }

  .quick-test-button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Results Panel */
  .results-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .results-controls {
    display: flex;
    gap: 1rem;
  }

  .control-button {
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

  .control-button:hover {
    background: #4ecdc4;
    color: #0a0a0a;
  }

  .results-stats {
    display: flex;
    gap: 2rem;
    margin-bottom: 2rem;
    justify-content: center;
  }

  .stat-item {
    text-align: center;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    min-width: 120px;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4ecdc4;
    margin-bottom: 0.5rem;
  }

  .stat-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .results-list {
    max-height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .result-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid;
  }

  .result-item.passed {
    background: rgba(0, 255, 65, 0.1);
    border-color: #00ff41;
  }

  .result-item.failed {
    background: rgba(255, 107, 107, 0.1);
    border-color: #ff6b6b;
  }

  .result-status {
    font-size: 1.2rem;
  }

  .result-content {
    flex: 1;
  }

  .result-test {
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .result-message {
    font-size: 0.9rem;
    opacity: 0.8;
    margin-bottom: 0.5rem;
  }

  .result-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    opacity: 0.7;
  }

  /* Console Panel */
  .console-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
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
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 3rem;
  }

  .back-button,
  .home-button {
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

  .back-button:hover,
  .home-button:hover {
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
    .test-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .page-title {
      font-size: 2rem;
    }

    .test-grid {
      grid-template-columns: 1fr;
    }

    .test-controls {
      flex-direction: column;
    }

    .results-stats {
      flex-direction: column;
      gap: 1rem;
    }

    .navigation-footer {
      flex-direction: column;
      gap: 1rem;
    }
  }
</style>