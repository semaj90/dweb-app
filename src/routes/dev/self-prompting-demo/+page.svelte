<script lang="ts">
  import { onMount } from 'svelte';

  // Self-prompting system state
  let isActive = $state(false);
  let currentIteration = $state(0);
  let maxIterations = $state(5);
  let prompts = $state([]);
  let responses = $state([]);
  let systemGoal = $state('Analyze legal documents for key clauses and risks');
  let refinementFocus = $state('accuracy');
  let isRunning = $state(false);
  let completionRate = $state(0);

  const refinementTypes = [
    { id: 'accuracy', label: 'Accuracy', description: 'Focus on improving factual correctness' },
    { id: 'depth', label: 'Depth', description: 'Increase analysis depth and detail' },
    { id: 'clarity', label: 'Clarity', description: 'Improve response clarity and readability' },
    { id: 'coverage', label: 'Coverage', description: 'Expand topic coverage and scope' },
    { id: 'efficiency', label: 'Efficiency', description: 'Optimize for speed and conciseness' }
  ];

  // Sample self-prompting chain
  const sampleChain = {
    goal: 'Analyze legal contract risks',
    iterations: [
      {
        prompt: 'Please analyze this legal contract for potential risks and important clauses.',
        response: 'I\'ve identified several key areas: liability clauses, termination conditions, and intellectual property rights. However, I should be more specific about risk levels.',
        refinement: 'Be more specific about risk severity levels and provide concrete examples.'
      },
      {
        prompt: 'Analyze this legal contract for potential risks, categorizing each risk by severity (High/Medium/Low) and providing specific examples from typical contracts.',
        response: 'HIGH RISK: Unlimited liability clauses, automatic renewal without notice. MEDIUM RISK: Vague termination conditions, broad non-compete terms. LOW RISK: Standard confidentiality clauses. Should also include mitigation strategies.',
        refinement: 'Add practical mitigation strategies for each identified risk level.'
      },
      {
        prompt: 'Analyze this legal contract for potential risks, categorizing by severity (High/Medium/Low) with examples, and provide specific mitigation strategies for each risk level.',
        response: 'HIGH RISK: Unlimited liability (Mitigation: Negotiate liability caps), Automatic renewal (Add termination notice requirements). MEDIUM RISK: Vague terms (Request specific definitions), Non-compete overreach (Limit scope and duration). LOW RISK: Standard clauses (Standard legal review sufficient).',
        refinement: 'Perfect! This response provides comprehensive risk analysis with actionable mitigation strategies.'
      }
    ]
  };

  async function startSelfPrompting() {
    if (!systemGoal.trim()) return;
    
    isRunning = true;
    isActive = true;
    currentIteration = 0;
    prompts = [];
    responses = [];
    completionRate = 0;

    // Initial prompt
    let currentPrompt = `Goal: ${systemGoal}\n\nPlease provide your initial analysis focusing on ${refinementFocus}.`;
    
    for (let i = 0; i < maxIterations; i++) {
      currentIteration = i + 1;
      completionRate = (i / maxIterations) * 100;
      
      // Add current prompt
      prompts = [...prompts, {
        id: i,
        iteration: i + 1,
        text: currentPrompt,
        timestamp: new Date().toLocaleTimeString(),
        focus: refinementFocus
      }];

      // Simulate AI processing
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Generate response (simulated)
      const response = generateSimulatedResponse(i, systemGoal, refinementFocus);
      responses = [...responses, {
        id: i,
        iteration: i + 1,
        text: response.content,
        timestamp: new Date().toLocaleTimeString(),
        quality: response.quality,
        improvements: response.improvements
      }];

      // Generate next prompt if not final iteration
      if (i < maxIterations - 1) {
        currentPrompt = generateRefinedPrompt(response, systemGoal, refinementFocus);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    completionRate = 100;
    isRunning = false;
  }

  function generateSimulatedResponse(iteration: number, goal: string, focus: string) {
    const baseResponses = {
      0: {
        content: `Initial analysis of "${goal}" shows several key areas that need attention. The current approach identifies basic patterns but lacks specificity and depth.`,
        quality: 60,
        improvements: ['Add specific examples', 'Increase analysis depth', 'Provide actionable insights']
      },
      1: {
        content: `Refined analysis with specific examples: Key risk areas include [A], [B], and [C]. Each area shows different severity levels. However, the analysis could benefit from more structured categorization.`,
        quality: 75,
        improvements: ['Implement structured categorization', 'Add severity scoring', 'Include mitigation strategies']
      },
      2: {
        content: `Structured analysis with severity categorization: HIGH (Score: 8-10): Critical liability issues, automatic renewals. MEDIUM (Score: 5-7): Vague termination clauses, IP ownership disputes. LOW (Score: 1-4): Standard boilerplate terms.`,
        quality: 85,
        improvements: ['Add mitigation strategies', 'Include precedent references', 'Enhance practical guidance']
      },
      3: {
        content: `Comprehensive analysis with mitigation strategies: HIGH RISKS with solutions: 1) Liability caps negotiation, 2) Notice period requirements. MEDIUM RISKS with solutions: 1) Definition clarifications, 2) Scope limitations. Includes relevant case law references.`,
        quality: 92,
        improvements: ['Fine-tune presentation format', 'Add cross-references', 'Optimize for specific use cases']
      },
      4: {
        content: `Optimized comprehensive legal risk analysis: Executive Summary + Detailed Risk Matrix + Actionable Mitigation Strategies + Legal Precedent References + Implementation Timeline. Tailored for ${focus} optimization with measurable outcomes.`,
        quality: 96,
        improvements: ['Perfect execution', 'All requirements met', 'Ready for production use']
      }
    };

    return baseResponses[iteration] || baseResponses[4];
  }

  function generateRefinedPrompt(previousResponse: any, goal: string, focus: string): string {
    const refinementSuggestions = {
      accuracy: 'be more factually precise and cite specific legal standards',
      depth: 'provide more detailed analysis and explore edge cases',
      clarity: 'use clearer language and better structure the response',
      coverage: 'expand the scope to cover additional relevant areas',
      efficiency: 'be more concise while maintaining comprehensive coverage'
    };

    return `Based on the previous response, please refine your analysis of "${goal}" to ${refinementSuggestions[focus]}. Address the following improvements: ${previousResponse.improvements.join(', ')}.`;
  }

  function stopSelfPrompting() {
    isRunning = false;
    isActive = false;
  }

  function clearHistory() {
    prompts = [];
    responses = [];
    currentIteration = 0;
    completionRate = 0;
    isActive = false;
  }

  function loadSampleChain() {
    systemGoal = sampleChain.goal;
    prompts = sampleChain.iterations.map((iter, index) => ({
      id: index,
      iteration: index + 1,
      text: iter.prompt,
      timestamp: `Sample ${index + 1}`,
      focus: 'accuracy'
    }));
    responses = sampleChain.iterations.map((iter, index) => ({
      id: index,
      iteration: index + 1,
      text: iter.response,
      timestamp: `Sample ${index + 1}`,
      quality: 60 + (index * 15),
      improvements: [iter.refinement]
    }));
    currentIteration = sampleChain.iterations.length;
    isActive = true;
    completionRate = 100;
  }

  function exportChain() {
    const data = {
      timestamp: new Date().toISOString(),
      goal: systemGoal,
      refinementFocus,
      iterations: currentIteration,
      prompts,
      responses,
      completionRate
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `self-prompting-chain-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<svelte:head>
  <title>Self-Prompting AI Demo</title>
</svelte:head>

<div class="self-prompting-demo">
  <div class="demo-header">
    <h1 class="page-title">
      <span class="title-icon">🔄</span>
      SELF-PROMPTING AI SYSTEM
    </h1>
    <div class="system-status">
      <span class="status-label">STATUS:</span>
      <span class="status-value {isActive ? 'active' : 'inactive'}">
        {isRunning ? 'RUNNING' : isActive ? 'ACTIVE' : 'INACTIVE'}
      </span>
    </div>
  </div>

  <div class="demo-grid">
    <!-- Configuration Panel -->
    <section class="config-panel">
      <h2 class="section-title">SYSTEM CONFIGURATION</h2>
      
      <div class="config-group">
        <label class="config-label">System Goal</label>
        <textarea
          bind:value={systemGoal}
          class="goal-input"
          rows="3"
          placeholder="Enter the main objective for self-prompting analysis..."
        ></textarea>
      </div>

      <div class="config-group">
        <label class="config-label">Refinement Focus</label>
        <select bind:value={refinementFocus} class="focus-selector">
          {#each refinementTypes as type}
            <option value={type.id}>{type.label} - {type.description}</option>
          {/each}
        </select>
      </div>

      <div class="config-group">
        <label class="config-label">Max Iterations</label>
        <input
          type="range"
          min="3"
          max="10"
          bind:value={maxIterations}
          class="iteration-slider"
        />
        <div class="iteration-value">{maxIterations} iterations</div>
      </div>

      <div class="control-buttons">
        <button
          class="control-button start-button {isRunning ? 'running' : ''}"
          onclick={startSelfPrompting}
          disabled={isRunning || !systemGoal.trim()}
        >
          <div class="button-icon">{isRunning ? '⏸️' : '▶️'}</div>
          <div class="button-text">
            {isRunning ? 'PROCESSING...' : 'START SELF-PROMPTING'}
          </div>
        </button>

        <button
          class="control-button stop-button"
          onclick={stopSelfPrompting}
          disabled={!isRunning}
        >
          <div class="button-icon">⏹️</div>
          <div class="button-text">STOP</div>
        </button>

        <button
          class="control-button sample-button"
          onclick={loadSampleChain}
          disabled={isRunning}
        >
          <div class="button-icon">📋</div>
          <div class="button-text">LOAD SAMPLE</div>
        </button>
      </div>
    </section>

    <!-- Progress Monitor -->
    <section class="progress-panel">
      <h2 class="section-title">PROGRESS MONITOR</h2>
      
      <div class="progress-stats">
        <div class="stat-item">
          <div class="stat-value">{currentIteration}</div>
          <div class="stat-label">CURRENT ITERATION</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{maxIterations}</div>
          <div class="stat-label">MAX ITERATIONS</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{Math.round(completionRate)}%</div>
          <div class="stat-label">COMPLETION</div>
        </div>
      </div>

      <div class="progress-bar">
        <div class="progress-fill" style="width: {completionRate}%"></div>
      </div>

      <div class="progress-info">
        <div class="info-item">
          <span class="info-label">GOAL:</span>
          <span class="info-value">{systemGoal || 'Not set'}</span>
        </div>
        <div class="info-item">
          <span class="info-label">FOCUS:</span>
          <span class="info-value">{refinementTypes.find(r => r.id === refinementFocus)?.label || 'Accuracy'}</span>
        </div>
      </div>
    </section>

    <!-- Chain Visualization -->
    <section class="chain-panel">
      <div class="chain-header">
        <h2 class="section-title">PROMPT-RESPONSE CHAIN</h2>
        <div class="chain-controls">
          <button class="control-button" onclick={clearHistory}>
            🗑️ CLEAR
          </button>
          <button class="control-button" onclick={exportChain}>
            📥 EXPORT
          </button>
        </div>
      </div>

      <div class="chain-container">
        {#if prompts.length === 0}
          <div class="empty-chain">
            <div class="empty-icon">🔄</div>
            <div class="empty-text">No prompting chain yet</div>
            <div class="empty-subtext">Start self-prompting to see the iterative improvement process</div>
          </div>
        {:else}
          {#each prompts as prompt, index}
            <div class="chain-item">
              <div class="iteration-header">
                <div class="iteration-number">ITERATION {prompt.iteration}</div>
                <div class="iteration-time">{prompt.timestamp}</div>
              </div>

              <!-- Prompt -->
              <div class="prompt-section">
                <div class="section-label">PROMPT</div>
                <div class="prompt-content">{prompt.text}</div>
              </div>

              <!-- Response -->
              {#if responses[index]}
                <div class="response-section">
                  <div class="section-label">
                    RESPONSE
                    <div class="quality-badge">
                      Quality: {responses[index].quality}%
                    </div>
                  </div>
                  <div class="response-content">{responses[index].text}</div>
                  
                  {#if responses[index].improvements}
                    <div class="improvements-section">
                      <div class="improvements-label">IDENTIFIED IMPROVEMENTS:</div>
                      <div class="improvements-list">
                        {#each responses[index].improvements as improvement}
                          <div class="improvement-item">• {improvement}</div>
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}

              {#if index < prompts.length - 1}
                <div class="iteration-arrow">
                  <div class="arrow-line"></div>
                  <div class="arrow-head">▼</div>
                  <div class="arrow-text">REFINING...</div>
                </div>
              {/if}
            </div>
          {/each}
        {/if}
      </div>
    </section>

    <!-- Analytics Dashboard -->
    <section class="analytics-panel">
      <h2 class="section-title">PERFORMANCE ANALYTICS</h2>
      
      {#if responses.length > 0}
        <div class="analytics-grid">
          <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-content">
              <div class="metric-value">
                {responses.length > 1 ? '+' : ''}{responses.length > 1 ? (responses[responses.length - 1].quality - responses[0].quality) : responses[0].quality}%
              </div>
              <div class="metric-label">Quality Improvement</div>
            </div>
          </div>

          <div class="metric-card">
            <div class="metric-icon">🔄</div>
            <div class="metric-content">
              <div class="metric-value">{responses.length}</div>
              <div class="metric-label">Total Iterations</div>
            </div>
          </div>

          <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-content">
              <div class="metric-value">{responses.length > 0 ? responses[responses.length - 1].quality : 0}%</div>
              <div class="metric-label">Final Quality</div>
            </div>
          </div>

          <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-content">
              <div class="metric-value">{refinementTypes.find(r => r.id === refinementFocus)?.label || 'N/A'}</div>
              <div class="metric-label">Focus Area</div>
            </div>
          </div>
        </div>

        <!-- Quality Chart -->
        <div class="quality-chart">
          <div class="chart-title">QUALITY PROGRESSION</div>
          <div class="chart-container">
            {#each responses as response, index}
              <div class="chart-bar">
                <div class="bar-fill" style="height: {response.quality}%">
                  <div class="bar-label">{response.quality}%</div>
                </div>
                <div class="bar-iteration">I{index + 1}</div>
              </div>
            {/each}
          </div>
        </div>
      {:else}
        <div class="empty-analytics">
          <div class="empty-icon">📊</div>
          <div class="empty-text">No analytics data yet</div>
          <div class="empty-subtext">Run a self-prompting session to see performance metrics</div>
        </div>
      {/if}
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
  .self-prompting-demo {
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
    border: 2px solid #fb7185;
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
    background: linear-gradient(45deg, #fb7185, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px #fb7185);
    animation: rotate 3s linear infinite;
  }

  .system-status {
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

  .status-value.inactive {
    background: rgba(255, 107, 107, 0.2);
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }

  .status-value.active {
    background: rgba(0, 255, 65, 0.2);
    color: #00ff41;
    border: 1px solid #00ff41;
  }

  .demo-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #fb7185;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #fb7185;
  }

  /* Configuration Panel */
  .config-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(251, 113, 133, 0.3);
  }

  .config-group {
    margin-bottom: 2rem;
  }

  .config-label {
    display: block;
    font-size: 0.9rem;
    color: #f0f0f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }

  .goal-input {
    width: 100%;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(251, 113, 133, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    border-radius: 4px;
    resize: vertical;
  }

  .goal-input:focus {
    outline: none;
    border-color: #fb7185;
    box-shadow: 0 0 15px rgba(251, 113, 133, 0.3);
  }

  .focus-selector {
    width: 100%;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(251, 113, 133, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    border-radius: 4px;
  }

  .iteration-slider {
    width: 100%;
    appearance: none;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    outline: none;
    border-radius: 3px;
    margin-bottom: 0.5rem;
  }

  .iteration-slider::-webkit-slider-thumb {
    appearance: none;
    width: 20px;
    height: 20px;
    background: #fb7185;
    cursor: pointer;
    border-radius: 50%;
  }

  .iteration-value {
    text-align: center;
    font-size: 0.9rem;
    color: #fb7185;
    font-weight: 600;
    padding: 0.3rem;
    background: rgba(251, 113, 133, 0.1);
    border-radius: 3px;
  }

  .control-buttons {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .control-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: linear-gradient(145deg, #fb7185, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 1rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 6px;
  }

  .control-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(251, 113, 133, 0.3);
  }

  .control-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .start-button.running {
    animation: pulse 1.5s ease-in-out infinite;
  }

  .stop-button {
    background: linear-gradient(145deg, #ff6b6b, #ffbf00);
  }

  .sample-button {
    background: linear-gradient(145deg, #4ecdc4, #a78bfa);
  }

  /* Progress Panel */
  .progress-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(251, 113, 133, 0.3);
  }

  .progress-stats {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2rem;
  }

  .stat-item {
    text-align: center;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    flex: 1;
    margin: 0 0.5rem;
  }

  .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #fb7185;
    margin-bottom: 0.5rem;
  }

  .stat-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .progress-bar {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #fb7185, #00ff41);
    transition: width 0.5s ease;
    border-radius: 4px;
  }

  .progress-info {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
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
    color: #fb7185;
    font-weight: 600;
  }

  /* Chain Panel */
  .chain-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(251, 113, 133, 0.3);
  }

  .chain-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .chain-controls {
    display: flex;
    gap: 1rem;
  }

  .chain-controls .control-button {
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    background: transparent;
    border: 2px solid #fb7185;
    color: #fb7185;
  }

  .chain-controls .control-button:hover {
    background: #fb7185;
    color: #0a0a0a;
  }

  .chain-container {
    max-height: 600px;
    overflow-y: auto;
  }

  .empty-chain {
    text-align: center;
    padding: 4rem;
    color: #f0f0f0;
    opacity: 0.6;
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    animation: rotate 3s linear infinite;
  }

  .empty-text {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .empty-subtext {
    font-size: 0.9rem;
    opacity: 0.8;
  }

  .chain-item {
    margin-bottom: 3rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .iteration-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(251, 113, 133, 0.3);
  }

  .iteration-number {
    font-size: 1.1rem;
    font-weight: 700;
    color: #fb7185;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .iteration-time {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.7;
  }

  .prompt-section,
  .response-section {
    margin-bottom: 1.5rem;
  }

  .section-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    font-weight: 600;
    color: #4ecdc4;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .quality-badge {
    background: rgba(78, 205, 196, 0.2);
    color: #4ecdc4;
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.7rem;
    border: 1px solid #4ecdc4;
  }

  .prompt-content,
  .response-content {
    background: rgba(0, 0, 0, 0.3);
    padding: 1.2rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.95rem;
    line-height: 1.5;
    color: #f0f0f0;
  }

  .improvements-section {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(251, 113, 133, 0.1);
    border-radius: 6px;
    border: 1px solid rgba(251, 113, 133, 0.3);
  }

  .improvements-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #fb7185;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .improvements-list {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .improvement-item {
    font-size: 0.85rem;
    color: #f0f0f0;
    opacity: 0.9;
  }

  .iteration-arrow {
    text-align: center;
    margin: 2rem 0;
    color: #fb7185;
  }

  .arrow-line {
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #fb7185, transparent);
    margin-bottom: 0.5rem;
  }

  .arrow-head {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
  }

  .arrow-text {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
  }

  /* Analytics Panel */
  .analytics-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(251, 113, 133, 0.3);
  }

  .analytics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
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
    color: #fb7185;
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

  .quality-chart {
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .chart-title {
    font-size: 1rem;
    font-weight: 600;
    color: #fb7185;
    margin-bottom: 1rem;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .chart-container {
    display: flex;
    align-items: end;
    gap: 1rem;
    height: 150px;
    padding: 1rem 0;
  }

  .chart-bar {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
  }

  .bar-fill {
    background: linear-gradient(to top, #fb7185, #00ff41);
    width: 100%;
    position: relative;
    border-radius: 3px 3px 0 0;
    min-height: 10px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    transition: height 0.5s ease;
  }

  .bar-label {
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
  }

  .bar-iteration {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    margin-top: 0.5rem;
  }

  .empty-analytics {
    text-align: center;
    padding: 3rem;
    color: #f0f0f0;
    opacity: 0.6;
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
    background: linear-gradient(145deg, #fb7185, #00ff41);
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
    box-shadow: 0 10px 25px rgba(251, 113, 133, 0.3);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
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

    .progress-stats {
      flex-direction: column;
      gap: 1rem;
    }

    .progress-stats .stat-item {
      margin: 0;
    }

    .analytics-grid {
      grid-template-columns: 1fr;
    }

    .navigation-footer {
      flex-direction: column;
      gap: 1rem;
    }
  }
</style>