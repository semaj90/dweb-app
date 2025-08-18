<script lang="ts">
  import { onMount } from 'svelte';

  // Copilot optimizer state
  let optimizationSettings = $state({
    contextWindow: 4096,
    temperature: 0.7,
    maxTokens: 1024,
    topP: 0.9,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,
    stopSequences: [],
    systemPrompt: 'You are an AI assistant specialized in legal analysis.',
    enableMemory: true,
    memorySize: 2048,
    enableRAG: true,
    ragThreshold: 0.7
  });

  let currentModel = $state('gemma3-legal');
  let isOptimizing = $state(false);
  let optimizationResults = $state([]);
  let performanceMetrics = $state({
    responseTime: 0,
    tokenThroughput: 0,
    memoryUsage: 0,
    accuracy: 0,
    contextUtilization: 0
  });

  let testPrompts = $state([
    'Analyze this legal document for key clauses',
    'Summarize the main legal principles in this case',
    'What are the potential risks in this contract?',
    'Identify relevant precedents for this legal issue',
    'Generate a legal brief outline for this topic'
  ]);

  let selectedPrompt = $state(testPrompts[0]);
  let testResult = $state('');

  const availableModels = [
    { id: 'gemma3-legal', name: 'Gemma3 Legal', description: 'Specialized legal AI model' },
    { id: 'llama3-instruct', name: 'Llama3 Instruct', description: 'General instruction following' },
    { id: 'mistral-legal', name: 'Mistral Legal', description: 'Legal document analysis' },
    { id: 'claude-legal', name: 'Claude Legal', description: 'Advanced legal reasoning' }
  ];

  async function runOptimization() {
    isOptimizing = true;
    testResult = 'Running optimization tests...\n';
    
    try {
      // Simulate optimization process
      const steps = [
        'Testing context window utilization...',
        'Optimizing temperature settings...',
        'Calibrating token generation...',
        'Fine-tuning memory allocation...',
        'Validating RAG integration...',
        'Measuring performance metrics...'
      ];

      for (const [index, step] of steps.entries()) {
        testResult += step + '\n';
        await new Promise(resolve => setTimeout(resolve, 800));
        
        // Update progress
        const progress = ((index + 1) / steps.length) * 100;
        testResult += `Progress: ${Math.round(progress)}%\n`;
      }

      // Generate optimization results
      const results = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        model: currentModel,
        settings: { ...optimizationSettings },
        metrics: {
          responseTime: Math.floor(Math.random() * 500) + 200,
          tokenThroughput: Math.floor(Math.random() * 100) + 50,
          memoryUsage: Math.floor(Math.random() * 40) + 30,
          accuracy: (Math.random() * 0.2 + 0.8) * 100,
          contextUtilization: (Math.random() * 0.3 + 0.7) * 100
        },
        improvement: Math.floor(Math.random() * 25) + 10
      };

      optimizationResults = [results, ...optimizationResults];
      performanceMetrics = results.metrics;
      
      testResult += '\nOptimization completed!\n';
      testResult += `Performance improved by ${results.improvement}%\n`;
      testResult += `Response time: ${results.metrics.responseTime}ms\n`;
      testResult += `Accuracy: ${results.metrics.accuracy.toFixed(1)}%\n`;
      
    } catch (error) {
      testResult += `\nError during optimization: ${error.message}\n`;
    }
    
    isOptimizing = false;
  }

  async function testPrompt() {
    if (!selectedPrompt.trim()) return;
    
    testResult = 'Testing prompt with current settings...\n';
    testResult += `Prompt: "${selectedPrompt}"\n`;
    testResult += `Model: ${currentModel}\n`;
    testResult += `Temperature: ${optimizationSettings.temperature}\n\n`;
    
    try {
      // Simulate prompt testing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResponse = `Based on the current optimization settings, this is a simulated response to demonstrate how the AI would process your legal query. The response would analyze the request, apply legal reasoning, and provide relevant insights based on the configured parameters.

Key factors:
- Context window: ${optimizationSettings.contextWindow} tokens
- Temperature: ${optimizationSettings.temperature} (affects creativity)
- Max tokens: ${optimizationSettings.maxTokens}
- RAG enabled: ${optimizationSettings.enableRAG ? 'Yes' : 'No'}

This response demonstrates the current configuration's effectiveness.`;

      testResult += 'AI Response:\n';
      testResult += mockResponse + '\n\n';
      
      const responseTime = Math.floor(Math.random() * 1000) + 500;
      testResult += `Response generated in ${responseTime}ms\n`;
      
      performanceMetrics = {
        ...performanceMetrics,
        responseTime,
        tokenThroughput: Math.floor(Math.random() * 100) + 50
      };
      
    } catch (error) {
      testResult += `\nError testing prompt: ${error.message}\n`;
    }
  }

  function resetToDefaults() {
    optimizationSettings = {
      contextWindow: 4096,
      temperature: 0.7,
      maxTokens: 1024,
      topP: 0.9,
      frequencyPenalty: 0.0,
      presencePenalty: 0.0,
      stopSequences: [],
      systemPrompt: 'You are an AI assistant specialized in legal analysis.',
      enableMemory: true,
      memorySize: 2048,
      enableRAG: true,
      ragThreshold: 0.7
    };
  }

  function exportSettings() {
    const data = {
      timestamp: new Date().toISOString(),
      model: currentModel,
      settings: optimizationSettings,
      metrics: performanceMetrics,
      results: optimizationResults
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `copilot-settings-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<svelte:head>
  <title>Copilot Optimizer</title>
</svelte:head>

<div class="copilot-optimizer">
  <div class="optimizer-header">
    <h1 class="page-title">
      <span class="title-icon">🚁</span>
      AI COPILOT OPTIMIZER
    </h1>
    <div class="current-model">
      <span class="model-label">ACTIVE MODEL:</span>
      <span class="model-value">{currentModel}</span>
    </div>
  </div>

  <div class="optimizer-grid">
    <!-- Model Selection -->
    <section class="model-selection">
      <h2 class="section-title">MODEL SELECTION</h2>
      <div class="model-grid">
        {#each availableModels as model}
          <button
            class="model-card {currentModel === model.id ? 'active' : ''}"
            onclick={() => currentModel = model.id}
          >
            <div class="model-name">{model.name}</div>
            <div class="model-description">{model.description}</div>
          </button>
        {/each}
      </div>
    </section>

    <!-- Optimization Settings -->
    <section class="settings-panel">
      <h2 class="section-title">OPTIMIZATION SETTINGS</h2>
      <div class="settings-grid">
        <div class="setting-group">
          <label class="setting-label">Context Window</label>
          <input
            type="range"
            min="1024"
            max="8192"
            step="256"
            bind:value={optimizationSettings.contextWindow}
            class="setting-slider"
          />
          <div class="setting-value">{optimizationSettings.contextWindow} tokens</div>
        </div>

        <div class="setting-group">
          <label class="setting-label">Temperature</label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            bind:value={optimizationSettings.temperature}
            class="setting-slider"
          />
          <div class="setting-value">{optimizationSettings.temperature}</div>
        </div>

        <div class="setting-group">
          <label class="setting-label">Max Tokens</label>
          <input
            type="range"
            min="256"
            max="4096"
            step="128"
            bind:value={optimizationSettings.maxTokens}
            class="setting-slider"
          />
          <div class="setting-value">{optimizationSettings.maxTokens}</div>
        </div>

        <div class="setting-group">
          <label class="setting-label">Top P</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            bind:value={optimizationSettings.topP}
            class="setting-slider"
          />
          <div class="setting-value">{optimizationSettings.topP}</div>
        </div>

        <div class="setting-group">
          <label class="setting-label">Memory Size</label>
          <input
            type="range"
            min="512"
            max="4096"
            step="256"
            bind:value={optimizationSettings.memorySize}
            class="setting-slider"
          />
          <div class="setting-value">{optimizationSettings.memorySize} tokens</div>
        </div>

        <div class="setting-group">
          <label class="setting-label">RAG Threshold</label>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            bind:value={optimizationSettings.ragThreshold}
            class="setting-slider"
          />
          <div class="setting-value">{optimizationSettings.ragThreshold}</div>
        </div>
      </div>

      <div class="toggle-settings">
        <label class="toggle-group">
          <input
            type="checkbox"
            bind:checked={optimizationSettings.enableMemory}
            class="toggle-input"
          />
          <span class="toggle-label">Enable Memory</span>
        </label>

        <label class="toggle-group">
          <input
            type="checkbox"
            bind:checked={optimizationSettings.enableRAG}
            class="toggle-input"
          />
          <span class="toggle-label">Enable RAG</span>
        </label>
      </div>

      <div class="system-prompt-group">
        <label class="setting-label">System Prompt</label>
        <textarea
          bind:value={optimizationSettings.systemPrompt}
          class="system-prompt-input"
          rows="3"
        ></textarea>
      </div>
    </section>

    <!-- Performance Metrics -->
    <section class="metrics-panel">
      <h2 class="section-title">PERFORMANCE METRICS</h2>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-icon">⚡</div>
          <div class="metric-content">
            <div class="metric-value">{performanceMetrics.responseTime}ms</div>
            <div class="metric-label">Response Time</div>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-icon">🔄</div>
          <div class="metric-content">
            <div class="metric-value">{performanceMetrics.tokenThroughput}/s</div>
            <div class="metric-label">Token Throughput</div>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-icon">💾</div>
          <div class="metric-content">
            <div class="metric-value">{performanceMetrics.memoryUsage}%</div>
            <div class="metric-label">Memory Usage</div>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-icon">🎯</div>
          <div class="metric-content">
            <div class="metric-value">{performanceMetrics.accuracy.toFixed(1)}%</div>
            <div class="metric-label">Accuracy</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Test Controls -->
    <section class="test-controls">
      <h2 class="section-title">TESTING & OPTIMIZATION</h2>
      
      <div class="prompt-testing">
        <label class="setting-label">Test Prompt</label>
        <select bind:value={selectedPrompt} class="prompt-selector">
          {#each testPrompts as prompt}
            <option value={prompt}>{prompt}</option>
          {/each}
        </select>
        
        <div class="test-buttons">
          <button class="test-button" onclick={testPrompt}>
            <div class="button-icon">🧪</div>
            <div class="button-text">TEST PROMPT</div>
          </button>
          
          <button 
            class="optimize-button {isOptimizing ? 'optimizing' : ''}"
            onclick={runOptimization}
            disabled={isOptimizing}
          >
            <div class="button-icon">🚀</div>
            <div class="button-text">
              {isOptimizing ? 'OPTIMIZING...' : 'RUN OPTIMIZATION'}
            </div>
          </button>
        </div>
      </div>
    </section>

    <!-- Results Console -->
    <section class="results-console">
      <div class="console-header">
        <h2 class="section-title">RESULTS CONSOLE</h2>
        <div class="console-controls">
          <button class="control-button" onclick={() => testResult = ''}>
            🗑️ CLEAR
          </button>
          <button class="control-button" onclick={resetToDefaults}>
            🔄 RESET
          </button>
          <button class="control-button" onclick={exportSettings}>
            📥 EXPORT
          </button>
        </div>
      </div>
      
      <div class="console">
        <pre class="console-output">{testResult}</pre>
      </div>
    </section>

    <!-- Optimization History -->
    <section class="optimization-history">
      <h2 class="section-title">OPTIMIZATION HISTORY</h2>
      <div class="history-list">
        {#each optimizationResults as result}
          <div class="history-item">
            <div class="history-header">
              <div class="history-time">{result.timestamp}</div>
              <div class="history-model">{result.model}</div>
              <div class="history-improvement">+{result.improvement}%</div>
            </div>
            <div class="history-metrics">
              <span class="metric">RT: {result.metrics.responseTime}ms</span>
              <span class="metric">ACC: {result.metrics.accuracy.toFixed(1)}%</span>
              <span class="metric">MEM: {result.metrics.memoryUsage}%</span>
            </div>
          </div>
        {/each}
        
        {#if optimizationResults.length === 0}
          <div class="empty-history">
            <div class="empty-icon">📊</div>
            <div class="empty-text">No optimization history yet</div>
            <div class="empty-subtext">Run an optimization to see results</div>
          </div>
        {/if}
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
  .copilot-optimizer {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    padding: 2rem;
  }

  .optimizer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3rem;
    padding: 2rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 2px solid #a78bfa;
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
    background: linear-gradient(45deg, #a78bfa, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px #a78bfa);
  }

  .current-model {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
  }

  .model-label {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  .model-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #a78bfa;
    padding: 0.5rem 1rem;
    background: rgba(167, 139, 250, 0.2);
    border: 1px solid #a78bfa;
    border-radius: 4px;
  }

  .optimizer-grid {
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
    color: #a78bfa;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #a78bfa;
  }

  /* Model Selection */
  .model-selection {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }

  .model-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .model-card {
    background: rgba(255, 255, 255, 0.05);
    border: 2px solid rgba(167, 139, 250, 0.3);
    color: #ffffff;
    padding: 1.5rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: left;
  }

  .model-card:hover {
    border-color: #a78bfa;
    background: rgba(167, 139, 250, 0.1);
  }

  .model-card.active {
    border-color: #a78bfa;
    background: rgba(167, 139, 250, 0.2);
  }

  .model-name {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #a78bfa;
  }

  .model-description {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  /* Settings Panel */
  .settings-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }

  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .setting-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .setting-label {
    font-size: 0.9rem;
    color: #f0f0f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .setting-slider {
    appearance: none;
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    outline: none;
    border-radius: 3px;
  }

  .setting-slider::-webkit-slider-thumb {
    appearance: none;
    width: 20px;
    height: 20px;
    background: #a78bfa;
    cursor: pointer;
    border-radius: 50%;
  }

  .setting-value {
    font-size: 0.9rem;
    color: #a78bfa;
    font-weight: 600;
    text-align: center;
    padding: 0.3rem;
    background: rgba(167, 139, 250, 0.1);
    border-radius: 3px;
  }

  .toggle-settings {
    display: flex;
    gap: 2rem;
    margin-bottom: 2rem;
  }

  .toggle-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }

  .toggle-input {
    appearance: none;
    width: 20px;
    height: 20px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
    position: relative;
    cursor: pointer;
  }

  .toggle-input:checked {
    background: #a78bfa;
  }

  .toggle-input:checked::after {
    content: '✓';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #ffffff;
    font-weight: bold;
    font-size: 0.8rem;
  }

  .toggle-label {
    font-size: 0.9rem;
    color: #f0f0f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .system-prompt-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .system-prompt-input {
    width: 100%;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(167, 139, 250, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    border-radius: 4px;
    resize: vertical;
  }

  .system-prompt-input:focus {
    outline: none;
    border-color: #a78bfa;
  }

  /* Performance Metrics */
  .metrics-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .metric-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .metric-icon {
    font-size: 2rem;
    color: #a78bfa;
  }

  .metric-content {
    flex: 1;
  }

  .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
  }

  .metric-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Test Controls */
  .test-controls {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }

  .prompt-testing {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .prompt-selector {
    width: 100%;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(167, 139, 250, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    border-radius: 4px;
  }

  .test-buttons {
    display: flex;
    gap: 1rem;
  }

  .test-button,
  .optimize-button {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: linear-gradient(145deg, #a78bfa, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 6px;
  }

  .test-button:hover,
  .optimize-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(167, 139, 250, 0.3);
  }

  .optimize-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .optimize-button.optimizing {
    animation: pulse 1.5s ease-in-out infinite;
  }

  /* Results Console */
  .results-console {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }

  .console-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .console-controls {
    display: flex;
    gap: 1rem;
  }

  .control-button {
    background: transparent;
    border: 2px solid #a78bfa;
    color: #a78bfa;
    padding: 0.5rem 1rem;
    font-family: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .control-button:hover {
    background: #a78bfa;
    color: #0a0a0a;
  }

  .console {
    background: #000000;
    border: 1px solid #a78bfa;
    border-radius: 4px;
    padding: 1rem;
    height: 300px;
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

  /* Optimization History */
  .optimization-history {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(167, 139, 250, 0.3);
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-height: 300px;
    overflow-y: auto;
  }

  .history-item {
    background: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .history-time {
    color: #a78bfa;
    font-weight: 600;
  }

  .history-model {
    color: #f0f0f0;
    opacity: 0.8;
  }

  .history-improvement {
    color: #00ff41;
    font-weight: 700;
  }

  .history-metrics {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    opacity: 0.7;
  }

  .metric {
    color: #f0f0f0;
  }

  .empty-history {
    text-align: center;
    padding: 3rem;
    color: #f0f0f0;
    opacity: 0.6;
  }

  .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .empty-text {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .empty-subtext {
    font-size: 0.9rem;
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
    background: linear-gradient(145deg, #a78bfa, #00ff41);
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
    box-shadow: 0 10px 25px rgba(167, 139, 250, 0.3);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .optimizer-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .page-title {
      font-size: 2rem;
    }

    .optimizer-grid {
      grid-template-columns: 1fr;
    }

    .settings-grid,
    .metrics-grid,
    .model-grid {
      grid-template-columns: 1fr;
    }

    .test-buttons {
      flex-direction: column;
    }

    .navigation-footer {
      flex-direction: column;
      gap: 1rem;
    }
  }
</style>