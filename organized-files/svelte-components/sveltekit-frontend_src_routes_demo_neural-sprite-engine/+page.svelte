<!-- NES-Inspired Neural Sprite Engine Demo -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import { fabric } from 'fabric';
  import { createNeuralSpriteEngine, createPerformanceStores } from '$lib/engines/neural-sprite-engine';
  import InteractionManager from '$lib/InteractionManager';
  import { Play, Pause, Square, Zap, Cpu, HardDrive, Gamepad2, Brain, Gauge } from 'lucide-svelte';

  let canvasElement: HTMLCanvasElement
  let fabricCanvas: fabric.Canvas;
  let neuralEngine: any
  let performanceStores: any
  let animationId: number

  // Demo state
  let isPlaying = false;
  let selectedDemo = 'legal-workflow';
  let aiPredictionEnabled = true;
  let gpuAcceleration = true;
  // Interaction manager for nearest-neighbor queries
  const interactionManager = new InteractionManager();
  let frameCount = 0;
  let lastFrameTime = performance.now();

  // Performance metrics
  let fps = 60;
  let gpuMemoryUsage = 0;
  let cacheEfficiency = 100;
  let aiConfidence = 0;

  // Demo scenarios
  const demoScenarios = [
    {
      id: 'legal-workflow',
      name: '⚖️ Legal Document Review',
      description: 'Simulate evidence annotation and document analysis workflow',
      sprites: ['idle', 'document_load', 'annotation_mode', 'ai_analysis', 'complete'],
      color: '#3b82f6'
    },
    {
      id: 'evidence-analysis',
      name: '🔍 Evidence Processing',
      description: 'AI-powered evidence categorization and cross-referencing',
      sprites: ['scanning', 'processing', 'categorizing', 'linking', 'validated'],
      color: '#059669'
    },
    {
      id: 'case-building',
      name: '📋 Case Construction',
      description: 'Dynamic case timeline building with predictive suggestions',
      sprites: ['timeline_start', 'evidence_add', 'pattern_detect', 'suggestion', 'case_complete'],
      color: '#dc2626'
    },
    {
      id: 'stress-test',
      name: '⚡ Performance Stress Test',
      description: 'High-frequency sprite switching to test NES-level performance',
      sprites: ['rapid_1', 'rapid_2', 'rapid_3', 'rapid_4', 'rapid_5'],
      color: '#7c3aed'
    }
  ];

  // AI prediction simulation
  let predictedNextStates = writable<string[]>([]);
  let currentPredictionAccuracy = 85;

  onMount(async () => {
    initializeNeuralSpriteEngine();
    setupDemoScenarios();
    startPerformanceMonitoring();
  });

  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
    if (neuralEngine) {
      neuralEngine.destroy();
    }
  });

  function initializeNeuralSpriteEngine() {
    // Initialize Fabric.js canvas with retro styling
    fabricCanvas = new fabric.Canvas(canvasElement, {
      width: 800,
      height: 600,
      backgroundColor: '#1a1a2e',
      selection: false
    });

    // Create neural sprite engine
    neuralEngine = createNeuralSpriteEngine(fabricCanvas);
    performanceStores = createPerformanceStores(neuralEngine);

    // Add retro grid background
    addRetroGrid();
  }

  function addRetroGrid() {
    const gridSize = 32; // NES-inspired 32x32 pixel grid
    const canvasWidth = fabricCanvas.getWidth();
    const canvasHeight = fabricCanvas.getHeight();

    // Create grid lines
    for (let i = 0; i <= canvasWidth; i += gridSize) {
      const line = new fabric.Line([i, 0, i, canvasHeight], {
        stroke: '#16213e',
        strokeWidth: 1,
        selectable: false,
        evented: false
      });
      fabricCanvas.add(line);
    }

    for (let i = 0; i <= canvasHeight; i += gridSize) {
      const line = new fabric.Line([0, i, canvasWidth, i], {
        stroke: '#16213e',
        strokeWidth: 1,
        selectable: false,
        evented: false
      });
      fabricCanvas.add(line);
    }

    fabricCanvas.renderAll();
  }

  async function setupDemoScenarios() {
    // Create sprite states for each demo scenario
    for (const scenario of demoScenarios) {
      for (let i = 0; i < scenario.sprites.length; i++) {
        const spriteName = scenario.sprites[i];

        // Clear canvas
        fabricCanvas.clear();
        addRetroGrid();

        // Create demo objects based on scenario
        await createScenarioObjects(scenario.id, spriteName, i);

        // Capture sprite state
        await neuralEngine.captureCurrentState(spriteName, [
          scenario.id,
          'demo',
          i === 0 ? 'initial' : 'transition',
          gpuAcceleration ? 'gpu_accelerated' : 'cpu_only'
        ]);
      }
    }

    // Start with idle state
    await neuralEngine.loadSprite('idle');
    // initialize InteractionManager with documents from backend (if available)
    try {
      const resp = await fetch('/api/v1/initial-data');
      if (resp.ok) {
        const data = await resp.json();
        const docs = (data.documents || []).map(d => ({ id: d.id, vector: d.vector }));
        await interactionManager.init(docs);
      }
    } catch (e) {
      // ignore
    }
  }

  async function createScenarioObjects(scenarioId: string, spriteName: string, frameIndex: number) {
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'];
    const baseColor = colors[frameIndex % colors.length];

    switch (scenarioId) {
      case 'legal-workflow':
        await createLegalWorkflowSprite(spriteName, frameIndex, baseColor);
        break;
      case 'evidence-analysis':
        await createEvidenceAnalysisSprite(spriteName, frameIndex, baseColor);
        break;
      case 'case-building':
        await createCaseBuildingSprite(spriteName, frameIndex, baseColor);
        break;
      case 'stress-test':
        await createStressTestSprite(spriteName, frameIndex, baseColor);
        break;
    }
  }

  async function createLegalWorkflowSprite(spriteName: string, frameIndex: number, color: string) {
    // Document representation
    const doc = new fabric.Rect({
      left: 100 + (frameIndex * 30),
      top: 100 + (frameIndex * 20),
      width: 200,
      height: 250,
      fill: color,
      stroke: '#fff',
      strokeWidth: 2,
      rx: 10,
      ry: 10
    });

    // Add document text
    const title = new fabric.Text(spriteName.replace('_', ' ').toUpperCase(), {
      left: 120,
      top: 120,
      fontSize: 16,
      fontFamily: 'monospace',
      fill: '#fff',
      fontWeight: 'bold'
    });

    // AI analysis indicator (pulsing effect based on frame)
    const aiIndicator = new fabric.Circle({
      left: 320 + Math.sin(frameIndex) * 20,
      top: 120 + Math.cos(frameIndex) * 20,
      radius: 15 + (frameIndex * 2),
      fill: '#00ff00',
      opacity: 0.7 + (frameIndex * 0.1)
    });

    fabricCanvas.add(doc, title, aiIndicator);
  }

  async function createEvidenceAnalysisSprite(spriteName: string, frameIndex: number, color: string) {
    // Evidence nodes
    for (let i = 0; i < 3 + frameIndex; i++) {
      const evidence = new fabric.Circle({
        left: 150 + (i * 80) + (frameIndex * 10),
        top: 200 + Math.sin(i + frameIndex) * 30,
        radius: 25,
        fill: color,
        stroke: '#fff',
        strokeWidth: 2
      });

      const label = new fabric.Text(`E${i + 1}`, {
        left: 145 + (i * 80) + (frameIndex * 10),
        top: 195 + Math.sin(i + frameIndex) * 30,
        fontSize: 14,
        fontFamily: 'monospace',
        fill: '#fff',
        fontWeight: 'bold'
      });

      fabricCanvas.add(evidence, label);
    }

    // Connection lines (showing AI analysis)
    if (frameIndex > 1) {
      const line = new fabric.Line([200, 220, 280, 220], {
        stroke: '#00ff00',
        strokeWidth: 3,
        strokeDashArray: [5, 5]
      });
      fabricCanvas.add(line);
    }
  }

  async function createCaseBuildingSprite(spriteName: string, frameIndex: number, color: string) {
    // Timeline representation
    const timeline = new fabric.Line([50, 300, 750, 300], {
      stroke: '#fff',
      strokeWidth: 4
    });

    fabricCanvas.add(timeline);

    // Timeline events
    for (let i = 0; i <= frameIndex; i++) {
      const event = new fabric.Rect({
        left: 100 + (i * 120),
        top: 250,
        width: 80,
        height: 100,
        fill: color,
        stroke: '#fff',
        strokeWidth: 2,
        opacity: 0.8 + (i * 0.05)
      });

      const eventText = new fabric.Text(`Event ${i + 1}`, {
        left: 110 + (i * 120),
        top: 290,
        fontSize: 12,
        fontFamily: 'monospace',
        fill: '#fff'
      });

      fabricCanvas.add(event, eventText);
    }
  }

  async function createStressTestSprite(spriteName: string, frameIndex: number, color: string) {
    // Rapid geometric patterns for stress testing
    const pattern = Math.floor(frameIndex / 2) % 4;

    for (let i = 0; i < 20; i++) {
      let shape;
      const x = (i % 10) * 70 + 50;
      const y = Math.floor(i / 10) * 200 + 100;

      switch (pattern) {
        case 0:
          shape = new fabric.Circle({
            left: x, top: y, radius: 20,
            fill: color, opacity: 0.7
          });
          break;
        case 1:
          shape = new fabric.Rect({
            left: x, top: y, width: 40, height: 40,
            fill: color, opacity: 0.7
          });
          break;
        case 2:
          shape = new fabric.Triangle({
            left: x, top: y, width: 40, height: 40,
            fill: color, opacity: 0.7
          });
          break;
        case 3:
          shape = new fabric.Polygon([
            {x: 0, y: -20}, {x: 20, y: 0}, {x: 0, y: 20}, {x: -20, y: 0}
          ], {
            left: x, top: y, fill: color, opacity: 0.7
          });
          break;
      }

      if (shape) fabricCanvas.add(shape);
    }
  }

  async function playDemo() {
    if (!neuralEngine) return;

    isPlaying = true;
    const scenario = demoScenarios.find(s => s.id === selectedDemo);
    if (!scenario) return;

    let currentFrame = 0;

    const playFrame = async () => {
      if (!isPlaying) return;

      const spriteName = scenario.sprites[currentFrame];

      // Simulate AI prediction
      if (aiPredictionEnabled) {
        simulateAIPrediction(scenario, currentFrame);
      }

      // Log user activity for learning
      neuralEngine.logUserActivity('demo_playback', {
        scenario: selectedDemo,
        frame: currentFrame,
        timestamp: Date.now(),
        gpuAccelerated: gpuAcceleration
      });

      // Load sprite with neural engine
      const loadStart = performance.now();
      await neuralEngine.loadSprite(spriteName);
      const loadTime = performance.now() - loadStart;

      // Update metrics
      updatePerformanceMetrics(loadTime);

      currentFrame = (currentFrame + 1) % scenario.sprites.length;

      // Schedule next frame (60 FPS target like NES)
      setTimeout(() => {
        if (isPlaying) {
          animationId = requestAnimationFrame(playFrame);
        }
      }, 1000 / 60);
    };

    playFrame();
  }

  function simulateAIPrediction(scenario: any, currentFrame: number) {
    // Predict next 2-3 states based on current pattern
    const predictions = [];
    for (let i = 1; i <= 3; i++) {
      const nextIndex = (currentFrame + i) % scenario.sprites.length;
      predictions.push(scenario.sprites[nextIndex]);
    }

    predictedNextStates.set(predictions);

    // Simulate varying AI confidence
    aiConfidence = 70 + Math.random() * 25;
    currentPredictionAccuracy = Math.min(95, currentPredictionAccuracy + (Math.random() - 0.5) * 5);
  }

  function updatePerformanceMetrics(loadTime: number) {
    frameCount++;

    // Calculate FPS
    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    fps = Math.round(1000 / deltaTime);
    lastFrameTime = now;

    // Simulate GPU memory usage
    gpuMemoryUsage = Math.min(100, gpuMemoryUsage + Math.random() * 2 - 1);

    // Cache efficiency based on load time
    cacheEfficiency = loadTime < 1 ? 100 : Math.max(60, 100 - (loadTime * 10));
  }

  function stopDemo() {
    isPlaying = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  }

  function resetDemo() {
    stopDemo();
    frameCount = 0;
    gpuMemoryUsage = 0;
    cacheEfficiency = 100;
    aiConfidence = 0;
    predictedNextStates.set([]);
  }

  function startPerformanceMonitoring() {
    setInterval(() => {
      // Simulate real-time metrics updates
      if (isPlaying) {
        gpuMemoryUsage = Math.max(0, Math.min(100, gpuMemoryUsage + (Math.random() - 0.5) * 3));
      }
    }, 100);
  }

  // Reactive statements
  // keep selectedScenario and predictionList simple and compatible with existing code
  $: selectedScenario = demoScenarios.find(s => s.id === selectedDemo);
  $: predictionList = $predictedNextStates;

  // Run a nearest-neighbor query using interactionManager and highlight results
  async function runNNQuery() {
    try {
      if (!interactionManager || !interactionManager.built) return;
      // use first document vector as a query fallback
      const docs = interactionManager.documents || [];
      if (docs.length === 0) return;
      const q = docs[0].vector;
      const res = await interactionManager.queryNearest(q, 5);
      const indices = res.map(r => r.index);
      if (neuralEngine && typeof neuralEngine.highlightDocumentIndices === 'function') {
        neuralEngine.highlightDocumentIndices(indices);
      } else {
        // fallback: call window hook if present
        if (window && (window.__DEMO_HIGHLIGHT__ as any) && typeof (window.__DEMO_HIGHLIGHT__ as any) === 'function') {
          (window.__DEMO_HIGHLIGHT__ as any)(indices);
        }
      }
    } catch (e) {
      console.warn('runNNQuery failed', e);
    }
  }
</script>

<svelte:head>
  <title>Neural Sprite Engine Demo - NES-Inspired AI Graphics</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
  </style>
</svelte:head>

<div class="nes-demo-container">
  <!-- NES-Style Header -->
  <header class="nes-header">
    <div class="nes-title">
      <h1>🎮 NEURAL SPRITE ENGINE v1.0</h1>
      <p>NES-Inspired AI Graphics with GPU Acceleration</p>
    </div>
    <div class="nes-stats">
      <div class="stat-item">
        <Gauge class="stat-icon" size="16" />
        <span>FPS: {fps}</span>
      </div>
      <div class="stat-item">
        <Cpu class="stat-icon" size="16" />
        <span>GPU: {Math.round(gpuMemoryUsage)}%</span>
      </div>
      <div class="stat-item">
        <HardDrive class="stat-icon" size="16" />
        <span>Cache: {Math.round(cacheEfficiency)}%</span>
      </div>
    </div>
  </header>

  <div class="nes-main-content">
    <!-- Control Panel -->
    <aside class="nes-control-panel">
      <div class="control-section">
        <h3>🎯 Demo Scenarios</h3>
        <div class="scenario-selector">
          {#each demoScenarios as scenario}
            <button
              class="scenario-btn"
              class:active={selectedDemo === scenario.id}
              style="border-color: {scenario.color}"
              onclick={() => { selectedDemo = scenario.id; resetDemo(); }}
            >
              <div class="scenario-name">{scenario.name}</div>
              <div class="scenario-desc">{scenario.description}</div>
            </button>
          {/each}
        </div>
      </div>

      <div class="control-section">
        <h3>⚡ Performance Settings</h3>
        <div class="settings-grid">
          <label class="setting-item">
            <input type="checkbox" bind:checked={aiPredictionEnabled} />
            <Brain size="16" />
            AI Prediction
          </label>
          <label class="setting-item">
            <input type="checkbox" bind:checked={gpuAcceleration} />
            <Zap size="16" />
            GPU Acceleration
          </label>
        </div>
      </div>

      <div class="control-section">
        <h3>🎮 Playback Controls</h3>
        <div class="control-buttons">
          {#if !isPlaying}
            <button class="nes-btn primary" onclick={playDemo}>
              <Play size="16" />
              Play Demo
            </button>
          {:else}
            <button class="nes-btn secondary" onclick={stopDemo}>
              <Pause size="16" />
              Pause
            </button>
          {/if}
          <button class="nes-btn reset" onclick={resetDemo}>
            <Square size="16" />
            Reset
          </button>
          <button class="nes-btn" onclick={runNNQuery} title="Run a nearest-neighbor query and highlight results">
            🔍 Run NN Query
          </button>
        </div>
      </div>

      <!-- AI Prediction Panel -->
      {#if aiPredictionEnabled && predictionList.length > 0}
        <div class="control-section ai-section">
          <h3>🧠 AI Predictions</h3>
          <div class="prediction-accuracy">
            Accuracy: {Math.round(currentPredictionAccuracy)}%
          </div>
          <div class="predicted-states">
            {#each predictionList as state, i}
              <div class="predicted-state" style="opacity: {1 - (i * 0.3)}">
                {state.replace('_', ' ')}
              </div>
            {/each}
          </div>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: {aiConfidence}%"></div>
            <span class="confidence-text">{Math.round(aiConfidence)}% confident</span>
          </div>
        </div>
      {/if}
    </aside>

    <!-- Main Canvas Area -->
    <main class="nes-canvas-container">
      <div class="canvas-header">
        <h2>
          {#if selectedScenario}
            {selectedScenario.name} - {selectedScenario.description}
          {:else}
            Neural Sprite Demonstration
          {/if}
        </h2>
        <div class="canvas-metrics">
          <span>Frame: {frameCount}</span>
          <span>•</span>
          <span>Load Time: &lt;1ms</span>
          <span>•</span>
          <span class="performance-grade">
            Grade: {$performanceStores?.performanceGrade || 'S'}
          </span>
        </div>
      </div>

      <div class="canvas-wrapper">
        <canvas bind:this={canvasElement} class="nes-canvas"></canvas>

        <!-- Overlay indicators -->
        {#if isPlaying}
          <div class="canvas-overlay">
            <div class="playback-indicator">
              <div class="pulse-dot"></div>
              <span>NEURAL ENGINE ACTIVE</span>
            </div>
          </div>
        {/if}

        {#if gpuAcceleration}
          <div class="gpu-indicator">
            <Zap size="14" />
            GPU
          </div>
        {/if}
      </div>
    </main>
  </div>

  <!-- Performance Dashboard -->
  <footer class="nes-performance-dashboard">
    <div class="dashboard-section">
      <h4>🏆 NES Performance Metrics</h4>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value">{fps}</div>
          <div class="metric-label">FPS</div>
          <div class="metric-target">Target: 60</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{Math.round(cacheEfficiency)}%</div>
          <div class="metric-label">Cache Hit</div>
          <div class="metric-target">Target: >90%</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">&lt;1ms</div>
          <div class="metric-label">Load Time</div>
          <div class="metric-target">NES: 16ms</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{Math.round(gpuMemoryUsage)}%</div>
          <div class="metric-label">GPU Memory</div>
          <div class="metric-target">Max: 256MB</div>
        </div>
      </div>
    </div>

    <div class="dashboard-section">
      <h4>🧠 AI Intelligence Stats</h4>
      <div class="ai-stats">
        <div class="stat-row">
          <span>Prediction Model:</span>
          <span class="stat-value">Gemma3-Legal v2.0</span>
        </div>
        <div class="stat-row">
          <span>Learning Rate:</span>
          <span class="stat-value">Real-time Adaptive</span>
        </div>
        <div class="stat-row">
          <span>Pattern Recognition:</span>
          <span class="stat-value">{Math.round(currentPredictionAccuracy)}% Accuracy</span>
        </div>
        <div class="stat-row">
          <span>Cache Intelligence:</span>
          <span class="stat-value">Active Learning</span>
        </div>
      </div>
    </div>
  </footer>
</div>

<style>
  /* @unocss-include */
  .nes-demo-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: #00ff41;
    font-family: 'Press Start 2P', monospace;
    padding: 1rem;
    overflow-x: auto
  }

  .nes-header {
    display: flex
    justify-content: space-between;
    align-items: center
    padding: 1rem 2rem;
    background: rgba(0, 0, 0, 0.8);
    border: 2px solid #00ff41;
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
  }

  .nes-title h1 {
    font-size: 1.5rem;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 0 10px #00ff41;
    animation: glow 2s ease-in-out infinite alternate;
  }

  .nes-title p {
    font-size: 0.7rem;
    margin: 0;
    opacity: 0.8;
  }

  .nes-stats {
    display: flex
    gap: 1.5rem;
  }

  .stat-item {
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(0, 255, 65, 0.1);
    border: 1px solid #00ff41;
    border-radius: 4px;
    font-size: 0.7rem;
  }

  .stat-icon {
    color: #00ff41;
  }

  .nes-main-content {
    display: grid
    grid-template-columns: 350px 1fr;
    gap: 1rem;
    height: 700px;
  }

  .nes-control-panel {
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff41;
    border-radius: 8px;
    padding: 1rem;
    overflow-y: auto
    box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.1);
  }

  .control-section {
    margin-bottom: 2rem;
  }

  .control-section h3 {
    font-size: 0.8rem;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #00ff41;
  }

  .scenario-selector {
    display: flex
    flex-direction: column
    gap: 0.5rem;
  }

  .scenario-btn {
    background: rgba(0, 0, 0, 0.7);
    border: 2px solid #333;
    border-radius: 6px;
    padding: 1rem;
    color: #00ff41;
    font-family: inherit
    font-size: 0.6rem;
    cursor: pointer
    transition: all 0.3s ease;
    text-align: left
  }

  .scenario-btn:hover {
    background: rgba(0, 255, 65, 0.1);
    border-color: #00ff41;
    transform: translateX(4px);
  }

  .scenario-btn.active {
    background: rgba(0, 255, 65, 0.2);
    border-color: #00ff41;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.3);
  }

  .scenario-name {
    font-weight: bold
    margin-bottom: 0.5rem;
  }

  .scenario-desc {
    opacity: 0.7;
    line-height: 1.3;
  }

  .settings-grid {
    display: flex
    flex-direction: column
    gap: 0.8rem;
  }

  .setting-item {
    display: flex
    align-items: center
    gap: 0.8rem;
    font-size: 0.7rem;
    cursor: pointer
  }

  .setting-item input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: #00ff41;
  }

  .control-buttons {
    display: flex
    gap: 0.5rem;
    flex-wrap: wrap
  }

  .nes-btn {
    background: rgba(0, 255, 65, 0.1);
    border: 2px solid #00ff41;
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
    color: #00ff41;
    font-family: inherit
    font-size: 0.6rem;
    cursor: pointer
    display: flex
    align-items: center
    gap: 0.5rem;
    transition: all 0.2s ease;
  }

  .nes-btn:hover {
    background: rgba(0, 255, 65, 0.2);
    transform: scale(1.05);
  }

  .nes-btn:active {
    transform: scale(0.95);
  }

  .nes-btn.primary {
    background: rgba(0, 255, 65, 0.2);
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
  }

  .nes-btn.secondary {
    border-color: #ffaa00;
    color: #ffaa00;
    background: rgba(255, 170, 0, 0.1);
  }

  .nes-btn.reset {
    border-color: #ff4444;
    color: #ff4444;
    background: rgba(255, 68, 68, 0.1);
  }

  .ai-section {
    background: rgba(0, 100, 255, 0.05);
    border: 1px solid rgba(0, 100, 255, 0.3);
    border-radius: 6px;
    padding: 1rem;
  }

  .prediction-accuracy {
    font-size: 0.6rem;
    margin-bottom: 0.8rem;
    color: #00aaff;
  }

  .predicted-states {
    display: flex
    flex-direction: column
    gap: 0.3rem;
    margin-bottom: 1rem;
  }

  .predicted-state {
    background: rgba(0, 100, 255, 0.1);
    border: 1px solid rgba(0, 100, 255, 0.3);
    border-radius: 3px;
    padding: 0.5rem;
    font-size: 0.6rem;
    text-transform: capitalize
  }

  .confidence-bar {
    position: relative
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid #00aaff;
    border-radius: 10px;
    height: 20px;
    overflow: hidden
  }

  .confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #00aaff, #00ff88);
    transition: width 0.5s ease;
  }

  .confidence-text {
    position: absolute
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 0.5rem;
    color: white
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
  }

  .nes-canvas-container {
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff41;
    border-radius: 8px;
    padding: 1rem;
    display: flex
    flex-direction: column
  }

  .canvas-header {
    display: flex
    justify-content: space-between;
    align-items: center
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #00ff41;
  }

  .canvas-header h2 {
    font-size: 0.8rem;
    margin: 0;
  }

  .canvas-metrics {
    display: flex
    gap: 0.8rem;
    font-size: 0.6rem;
    opacity: 0.8;
  }

  .performance-grade {
    color: #00ff88;
    font-weight: bold
  }

  .canvas-wrapper {
    position: relative
    flex: 1;
    display: flex
    justify-content: center
    align-items: center
  }

  .nes-canvas {
    border: 2px solid #333;
    border-radius: 4px;
    background: #1a1a2e;
    box-shadow: 0 0 30px rgba(0, 255, 65, 0.2);
  }

  .canvas-overlay {
    position: absolute
    top: 1rem;
    left: 1rem;
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid #00ff41;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    display: flex
    align-items: center
    gap: 0.5rem;
    font-size: 0.6rem;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background: #00ff41;
    border-radius: 50%;
    animation: pulse 1s ease-in-out infinite;
  }

  .gpu-indicator {
    position: absolute
    top: 1rem;
    right: 1rem;
    background: rgba(255, 170, 0, 0.1);
    border: 1px solid #ffaa00;
    border-radius: 4px;
    padding: 0.3rem 0.6rem;
    color: #ffaa00;
    font-size: 0.6rem;
    display: flex
    align-items: center
    gap: 0.3rem;
  }

  .nes-performance-dashboard {
    margin-top: 1rem;
    background: rgba(0, 0, 0, 0.9);
    border: 2px solid #00ff41;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    display: grid
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }

  .dashboard-section h4 {
    font-size: 0.8rem;
    margin: 0 0 1rem 0;
    color: #00ff88;
  }

  .metrics-grid {
    display: grid
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
  }

  .metric-card {
    background: rgba(0, 255, 65, 0.05);
    border: 1px solid #00ff41;
    border-radius: 6px;
    padding: 1rem;
    text-align: center
  }

  .metric-value {
    font-size: 1.2rem;
    font-weight: bold
    color: #00ff88;
    margin-bottom: 0.3rem;
  }

  .metric-label {
    font-size: 0.6rem;
    margin-bottom: 0.3rem;
  }

  .metric-target {
    font-size: 0.5rem;
    opacity: 0.6;
  }

  .ai-stats {
    display: flex
    flex-direction: column
    gap: 0.8rem;
  }

  .stat-row {
    display: flex
    justify-content: space-between;
    font-size: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(0, 255, 65, 0.2);
  }

  .stat-value {
    color: #00ff88;
    font-weight: bold
  }

  /* Animations */
  @keyframes glow {
    from {
      text-shadow: 0 0 10px #00ff41;
    }
    to {
      text-shadow: 0 0 20px #00ff41, 0 0 30px #00ff41;
    }
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.2);
    }
  }

  /* Responsive Design */
  @media (max-width: 1200px) {
    .nes-main-content {
      grid-template-columns: 1fr;
      height: auto
    }

    .nes-performance-dashboard {
      grid-template-columns: 1fr;
    }

    .metrics-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 768px) {
    .nes-header {
      flex-direction: column
      gap: 1rem;
      text-align: center
    }

    .nes-stats {
      flex-wrap: wrap
      justify-content: center
    }

    .metrics-grid {
      grid-template-columns: 1fr;
    }
  }
</style>