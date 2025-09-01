<!-- YoRHa Advanced Command Interface - Complete 3D System -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import type { 
    SystemMetrics, 
    CommandResult, 
    YoRHaModule,
    HolographicData 
  } from '$lib/types/yorha-interface';
  
  // Core system stores
  const systemStatus = writable<'ONLINE' | 'DEGRADED' | 'OFFLINE'>('ONLINE');
  const powerLevel = writable<number>(98.7);
  const activeModules = writable<YoRHaModule[]>([]);
  const commandHistory = writable<CommandResult[]>([]);
  const holographicData = writable<HolographicData[]>([]);
  
  // 3D visualization elements
let canvas3D = $state<HTMLCanvasElement;
  let glContext: WebGLRenderingContext | null >(null);
let animationId = $state<number;
  
  // System metrics
  let metrics >(writable<SystemMetrics>({
    cpu_usage: 45.2,
    memory_usage: 67.8,
    gpu_utilization: 89.3,
    network_latency: 12,
    active_processes: 847,
    security_level: 'MAXIMUM',
    quantum_state: 'COHERENT',
    neural_activity: 94.6,
  }));
  
  // Command input
let commandInput = $state('');
let isProcessingCommand = $state(false);
  
  // Visual effects
let glitchActive = $state(false);
let scanlineOpacity = $state(0.3);
let hologramFlicker = $state(false);
  
  // YoRHa modules configuration
  const yorhaModules: YoRHaModule[] = [
    {
      id: 'legal-ai',
      name: 'LEGAL AI CORE',
      status: 'ACTIVE',
      power: 96.8,
      description: 'Advanced Legal Document Analysis',
      icon: 'gavel',
      color: '#00ff88'
    },
    {
      id: 'evidence-processor',
      name: 'EVIDENCE ANALYSIS',
      status: 'STANDBY',
      power: 78.2,
      description: 'Digital Evidence Processing Unit',
      icon: 'search',
      color: '#ff6b35'
    },
    {
      id: 'neural-network',
      name: 'NEURAL MATRIX',
      status: 'ACTIVE',
      power: 94.1,
      description: 'Machine Learning Core System',
      icon: 'brain',
      color: '#3b82f6'
    },
    {
      id: 'quantum-db',
      name: 'QUANTUM DATABASE',
      status: 'ACTIVE',
      power: 89.7,
      description: 'High-Speed Data Storage Matrix',
      icon: 'database',
      color: '#8b5cf6'
    },
    {
      id: 'security-grid',
      name: 'SECURITY GRID',
      status: 'MAXIMUM',
      power: 99.2,
      description: 'Perimeter Defense System',
      icon: 'shield',
      color: '#ef4444'
    },
    {
      id: 'comms-array',
      name: 'COMMUNICATIONS',
      status: 'ACTIVE',
      power: 87.4,
      description: 'Multi-Protocol Communication Hub',
      icon: 'radio',
      color: '#06b6d4'
    }
  ];
  
  onMount(() => {
    activeModules.set(yorhaModules);
    initializeHolographics();
    startSystemMonitoring();
    initWebGL();
    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  });
  
  function initializeHolographics() {
    const initialData: HolographicData[] = [
      {
        id: 'central-core',
        type: 'sphere',
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: 1.0,
        color: '#00ff88',
        opacity: 0.8,
        animation: 'rotate'
      },
      {
        id: 'data-streams',
        type: 'lines',
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: 1.0,
        color: '#3b82f6',
        opacity: 0.6,
        animation: 'flow'
      },
      {
        id: 'neural-nodes',
        type: 'points',
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: 1.0,
        color: '#8b5cf6',
        opacity: 0.9,
        animation: 'pulse'
      }
    ];
    holographicData.set(initialData);
  }
  
  function startSystemMonitoring() {
    setInterval(() => {
      metrics.update(current => ({
        ...current,
        cpu_usage: 40 + Math.random() * 20,
        memory_usage: 60 + Math.random() * 15,
        gpu_utilization: 85 + Math.random() * 10,
        network_latency: 8 + Math.random() * 8,
        neural_activity: 90 + Math.random() * 8,
      }));
      
      // Random glitch effects
      if (Math.random() < 0.05) {
        triggerGlitch();
      }
    }, 2000);
  }
  
  function initWebGL() {
    if (!canvas3D) return;
    
    glContext = canvas3D.getContext('webgl');
    if (!glContext) {
      console.warn('WebGL not supported, falling back to 2D rendering');
      return;
    }
    
    // Initialize WebGL rendering pipeline
    setupShaders();
    startRenderLoop();
  }
  
  function setupShaders() {
    if (!glContext) return;
    
    // Vertex shader for holographic effects
    const vertexShaderSource = `
      attribute vec4 position;
      attribute vec3 normal;
      uniform mat4 modelViewMatrix;
      uniform mat4 projectionMatrix;
      uniform float time;
      varying vec3 vNormal;
      varying float vGlow;
      
      void main() {
        vec4 pos = position;
        pos.y += sin(pos.x * 2.0 + time * 3.0) * 0.1;
        pos.x += cos(pos.z * 1.5 + time * 2.0) * 0.05;
        
        gl_Position = projectionMatrix * modelViewMatrix * pos;
        vNormal = normal;
        vGlow = abs(sin(time * 4.0 + position.x)) * 0.5 + 0.5;
      }
    `;
    
    // Fragment shader for cyber effects
    const fragmentShaderSource = `
      precision mediump float;
      uniform vec3 color;
      uniform float opacity;
      uniform float time;
      varying vec3 vNormal;
      varying float vGlow;
      
      void main() {
        vec3 finalColor = color * vGlow;
        finalColor += vec3(0.0, 1.0, 0.5) * abs(sin(time * 5.0)) * 0.3;
        
        float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 2.0);
        finalColor *= (1.0 + fresnel);
        
        gl_FragColor = vec4(finalColor, opacity * vGlow);
      }
    `;
    
    // Compile and link shaders (simplified implementation)
    console.log('YoRHa: WebGL shaders initialized');
  }
  
  function startRenderLoop() {
    const render = (timestamp: number) => {
      if (!glContext || !canvas3D) return;
      
      // Clear and setup viewport
      glContext.viewport(0, 0, canvas3D.width, canvas3D.height);
      glContext.clearColor(0.02, 0.05, 0.1, 1.0);
      glContext.clear(glContext.COLOR_BUFFER_BIT | glContext.DEPTH_BUFFER_BIT);
      
      // Render holographic elements
      renderHolographics(timestamp);
      
      animationId = requestAnimationFrame(render);
    };
    
    render(0);
  }
  
  function renderHolographics(timestamp: number) {
    // 3D holographic rendering implementation
    const time = timestamp * 0.001;
    
    // Update holographic data rotations
    holographicData.update(data => 
      data.map(item => ({
        ...item,
        rotation: {
          x: item.rotation.x + (item.animation === 'rotate' ? 0.01 : 0),
          y: item.rotation.y + (item.animation === 'rotate' ? 0.02 : 0),
          z: item.rotation.z + (item.animation === 'rotate' ? 0.005 : 0)
        }
      }))
    );
  }
  
  async function executeCommand(command: string) {
    if (isProcessingCommand) return;
    
    isProcessingCommand = true;
    commandInput = '';
    
    const result: CommandResult = {
      id: `cmd-${Date.now()}`,
      command,
      timestamp: new Date().toISOString(),
      status: 'PROCESSING',
      output: 'Executing command...',
      module: 'YORHA-CORE'
    };
    
    commandHistory.update(history => [result, ...history.slice(0, 9)]);
    
    try {
      // Route command to appropriate system
      const response = await routeCommand(command);
      
      result.status = 'SUCCESS';
      result.output = response.output;
      result.data = response.data;
      
    } catch (error) {
      result.status = 'ERROR';
      result.output = error instanceof Error ? error.message : 'Unknown error';
    } finally {
      isProcessingCommand = false;
      commandHistory.update(history => 
        history.map(cmd => cmd.id === result.id ? result : cmd)
      );
    }
  }
  
  async function routeCommand(command: string) {
    const cmd = command.toLowerCase().trim();
    
    if (cmd.startsWith('legal')) {
      return await executeLegalCommand(cmd);
    } else if (cmd.startsWith('analyze')) {
      return await executeAnalysisCommand(cmd);
    } else if (cmd.startsWith('search')) {
      return await executeSearchCommand(cmd);
    } else if (cmd.startsWith('system')) {
      return executeSystemCommand(cmd);
    } else if (cmd.startsWith('neural')) {
      return await executeNeuralCommand(cmd);
    } else {
      return executeHelpCommand(cmd);
    }
  }
  
  async function executeLegalCommand(cmd: string): Promise<any> {
    // Integration with legal AI services
    const response = await fetch('/api/v1/legal-ai/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: cmd, source: 'yorha-interface' })
    });
    
    if (response.ok) {
      const data = await response.json();
      return {
        output: `Legal analysis complete. ${data.summary || 'Analysis processed.'}`,
        data: data
      };
    } else {
      throw new Error('Legal AI system unavailable');
    }
  }
  
  async function executeAnalysisCommand(cmd: string): Promise<any> {
    return {
      output: `Analysis initiated: ${cmd.replace('analyze ', '')}`,
      data: { analysis_id: 'ANL-' + Date.now(), status: 'queued' }
    };
  }
  
  async function executeSearchCommand(cmd: string): Promise<any> {
    const query = cmd.replace('search ', '');
    return {
      output: `Searching database for: "${query}"`,
      data: { query, results_count: Math.floor(Math.random() * 50) + 1 }
    };
  }
  
  function executeSystemCommand(cmd: string) {
    if (cmd.includes('status')) {
      return {
        output: 'All systems operational. YoRHa interface running at optimal parameters.',
        data: $metrics
      };
    } else if (cmd.includes('modules')) {
      return {
        output: `${$activeModules.length} modules active`,
        data: $activeModules
      };
    } else {
      return {
        output: 'System command processed',
        data: { status: 'ok' }
      };
    }
  }
  
  async function executeNeuralCommand(cmd: string): Promise<any> {
    return {
      output: 'Neural network processing initiated',
      data: { neural_activity: $metrics.neural_activity }
    };
  }
  
  function executeHelpCommand(cmd: string) {
    return {
      output: `Available commands: LEGAL <query>, ANALYZE <target>, SEARCH <terms>, SYSTEM STATUS, NEURAL SCAN`,
      data: { commands: ['legal', 'analyze', 'search', 'system', 'neural'] }
    };
  }
  
  function triggerGlitch() {
    glitchActive = true;
    hologramFlicker = true;
    
    setTimeout(() => {
      glitchActive = false;
      hologramFlicker = false;
    }, 200);
  }
  
  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && commandInput.trim()) {
      executeCommand(commandInput.trim());
    }
  }
  
  function getStatusColor(status: string): string {
    switch (status) {
      case 'ACTIVE': return '#00ff88';
      case 'STANDBY': return '#ffa500';
      case 'MAXIMUM': return '#ff0066';
      case 'OFFLINE': return '#666';
      default: return '#00ff88';
    }
  }
</script>

<!-- YoRHa Interface Styles -->
<style>
  .yorha-container {
    @apply min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900;
    font-family: 'Courier New', 'Monaco', monospace;
    overflow: hidden;
    position: relative;
  }
  
  .scanlines {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      transparent 50%,
      rgba(0, 255, 136, 0.03) 50%
    );
    background-size: 100% 2px;
    animation: scanline-scroll 0.1s linear infinite;
    pointer-events: none;
    z-index: 1;
  }
  
  @keyframes scanline-scroll {
    0% { transform: translateY(0); }
    100% { transform: translateY(2px); }
  }
  
  .glitch-effect {
    animation: glitch 0.2s ease-in-out;
  }
  
  @keyframes glitch {
    0% { transform: translateX(0); filter: hue-rotate(0deg); }
    20% { transform: translateX(-2px); filter: hue-rotate(90deg); }
    40% { transform: translateX(2px); filter: hue-rotate(180deg); }
    60% { transform: translateX(-1px); filter: hue-rotate(270deg); }
    80% { transform: translateX(1px); filter: hue-rotate(360deg); }
    100% { transform: translateX(0); filter: hue-rotate(0deg); }
  }
  
  .cyber-border {
    border: 1px solid #00ff88;
    box-shadow: 
      0 0 5px rgba(0, 255, 136, 0.3),
      inset 0 0 5px rgba(0, 255, 136, 0.1);
  }
  
  .hologram-text {
    color: #00ff88;
    text-shadow: 0 0 10px currentColor;
    animation: hologram-flicker 2s ease-in-out infinite alternate;
  }
  
  @keyframes hologram-flicker {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.9; }
  }
  
  .module-panel {
    background: rgba(0, 20, 40, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 255, 136, 0.3);
    transition: all 0.3s ease;
  }
  
  .module-panel:hover {
    border-color: #00ff88;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    transform: translateY(-2px);
  }
  
  .command-terminal {
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid #00ff88;
    font-family: 'Courier New', monospace;
  }
  
  .command-input {
    background: transparent;
    border: none;
    color: #00ff88;
    font-family: inherit;
    outline: none;
    width: 100%;
  }
  
  .command-input::placeholder {
    color: rgba(0, 255, 136, 0.5);
  }
  
  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: status-pulse 2s ease-in-out infinite;
  }
  
  @keyframes status-pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
  
  .holographic-canvas {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    opacity: 0.6;
    z-index: 0;
  }
  
  .data-stream {
    position: absolute;
    width: 2px;
    height: 100px;
    background: linear-gradient(to bottom, transparent, #00ff88, transparent);
    animation: data-flow 3s linear infinite;
  }
  
  @keyframes data-flow {
    0% { transform: translateY(-100%); opacity: 0; }
    50% { opacity: 1; }
    100% { transform: translateY(100vh); opacity: 0; }
  }
</style>

<!-- YoRHa Command Interface -->
<div class="yorha-container" class:glitch-effect={glitchActive}>
  <!-- Scanline overlay -->
  <div class="scanlines" style="opacity: {scanlineOpacity}"></div>
  
  <!-- Background holographic canvas -->
  <canvas 
    bind:this={canvas3D}
    class="holographic-canvas"
    width="800" 
    height="600"
    style="filter: {hologramFlicker ? 'brightness(1.5) hue-rotate(180deg)' : 'brightness(1)'}"
  ></canvas>
  
  <!-- Animated data streams -->
  {#each Array(12) as _, i}
    <div 
      class="data-stream" 
      style="left: {5 + i * 8}%; animation-delay: {i * 0.3}s"
    ></div>
  {/each}
  
  <!-- Main interface content -->
  <div class="relative z-10 p-6 h-screen flex flex-col">
    
    <!-- Header -->
    <header class="flex justify-between items-center mb-8">
      <div>
        <h1 class="hologram-text text-4xl font-bold mb-2">YoRHa COMMAND INTERFACE</h1>
        <p class="text-cyan-400 text-sm">BUNKER UNIT 001 • LEGAL AI OPERATIONS CENTER</p>
      </div>
      
      <div class="flex items-center space-x-6">
        <div class="text-right">
          <div class="hologram-text text-2xl font-mono">{$powerLevel.toFixed(1)}%</div>
          <div class="text-cyan-400 text-xs">POWER LEVEL</div>
        </div>
        
        <div class="flex items-center space-x-2">
          <div 
            class="status-indicator"
            style="background-color: {getStatusColor($systemStatus)}"
          ></div>
          <span class="hologram-text font-mono">{$systemStatus}</span>
        </div>
      </div>
    </header>
    
    <!-- System modules grid -->
    <div class="grid grid-cols-3 gap-6 mb-8">
      {#each $activeModules as module}
        <div class="module-panel p-4 rounded-lg">
          <div class="flex justify-between items-start mb-3">
            <h3 class="hologram-text font-bold text-sm">{module.name}</h3>
            <div 
              class="status-indicator"
              style="background-color: {getStatusColor(module.status)}"
            ></div>
          </div>
          
          <p class="text-cyan-300 text-xs mb-3">{module.description}</p>
          
          <div class="flex justify-between items-center">
            <span class="text-white text-xs">{module.status}</span>
            <div class="text-right">
              <div class="hologram-text font-mono text-sm">{module.power.toFixed(1)}%</div>
              <div class="text-cyan-400 text-xs">PWR</div>
            </div>
          </div>
          
          <!-- Mini progress bar -->
          <div class="w-full bg-slate-700 h-1 rounded mt-2">
            <div 
              class="h-1 rounded"
              style="width: {module.power}%; background-color: {module.color}"
            ></div>
          </div>
        </div>
      {/each}
    </div>
    
    <!-- System metrics -->
    <div class="grid grid-cols-4 gap-4 mb-8">
      <div class="cyber-border p-3 rounded bg-black bg-opacity-50">
        <div class="hologram-text font-mono text-xl">{$metrics.cpu_usage.toFixed(1)}%</div>
        <div class="text-cyan-400 text-xs">CPU USAGE</div>
      </div>
      
      <div class="cyber-border p-3 rounded bg-black bg-opacity-50">
        <div class="hologram-text font-mono text-xl">{$metrics.memory_usage.toFixed(1)}%</div>
        <div class="text-cyan-400 text-xs">MEMORY</div>
      </div>
      
      <div class="cyber-border p-3 rounded bg-black bg-opacity-50">
        <div class="hologram-text font-mono text-xl">{$metrics.gpu_utilization.toFixed(1)}%</div>
        <div class="text-cyan-400 text-xs">GPU</div>
      </div>
      
      <div class="cyber-border p-3 rounded bg-black bg-opacity-50">
        <div class="hologram-text font-mono text-xl">{$metrics.neural_activity.toFixed(1)}%</div>
        <div class="text-cyan-400 text-xs">NEURAL</div>
      </div>
    </div>
    
    <!-- Command terminal -->
    <div class="flex-1 flex flex-col">
      <div class="command-terminal rounded-lg p-4 flex-1 flex flex-col">
        <div class="flex items-center mb-4">
          <span class="hologram-text text-sm mr-2">YORHA-CMD:</span>
          <input
            bind:value={commandInput}
            onkeypress={handleKeyPress}
            class="command-input flex-1"
            placeholder="Enter command..."
            disabled={isProcessingCommand}
          />
          {#if isProcessingCommand}
            <div class="hologram-text text-sm ml-2">PROCESSING...</div>
          {/if}
        </div>
        
        <!-- Command history -->
        <div class="flex-1 overflow-y-auto space-y-2">
          {#each $commandHistory as result}
            <div class="border-b border-cyan-900 pb-2">
              <div class="flex justify-between items-center mb-1">
                <span class="text-cyan-300 text-sm">$ {result.command}</span>
                <span 
                  class="text-xs px-2 py-1 rounded"
                  class:bg-green-900={result.status === 'SUCCESS'}
                  class:bg-red-900={result.status === 'ERROR'}
                  class:bg-yellow-900={result.status === 'PROCESSING'}
                  class:text-green-300={result.status === 'SUCCESS'}
                  class:text-red-300={result.status === 'ERROR'}
                  class:text-yellow-300={result.status === 'PROCESSING'}
                >
                  {result.status}
                </span>
              </div>
              <div class="text-white text-sm font-mono">{result.output}</div>
              <div class="text-xs text-gray-500 mt-1">
                [{result.module}] {new Date(result.timestamp).toLocaleTimeString()}
              </div>
            </div>
          {/each}
        </div>
      </div>
    </div>
    
    <!-- Footer -->
    <footer class="mt-4 text-center text-cyan-400 text-xs">
      <div>YoRHa Legal AI Interface v2.0 • Bunker Operations • Quantum Security Enabled</div>
      <div class="mt-1">Neural Activity: {$metrics.neural_activity.toFixed(1)}% • Security Level: {$metrics.security_level}</div>
    </footer>
  </div>
</div>