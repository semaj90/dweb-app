<script lang="ts">
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import { NESYoRHaHybrid3D, createNESButton, createNESContainer, createNESProgressBar, NES_YORHA_PALETTE } from '$lib/components/three/yorha-ui/NESYoRHaHybrid3D';
  import { nesCacheOrchestrator } from '$lib/services/nes-cache-orchestrator';
  import FinalFantasyModal from '$lib/components/ui/FinalFantasyModal.svelte';
  import FinalFantasyButton from '$lib/components/ui/FinalFantasyButton.svelte';
  import FinalFantasyContainer from '$lib/components/ui/FinalFantasyContainer.svelte';

  let canvasElement: HTMLCanvasElement;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let animationId: number;
  
  // UI State
  let isLoading = $state(false);
  let progress = $state(0);
  let currentDemo = $state('buttons');
  let cacheStats = $state<any>(null);
  let selectedComponent = $state<NESYoRHaHybrid3D | null>(null);

  // Final Fantasy Modal States
  let showYoRHaModal = $state(false);
  let showNESModal = $state(false);
  let showWebGLModal = $state(false);
  let showHybridModal = $state(false);

  // 3D Components
  let nesButtons: NESYoRHaHybrid3D[] = [];
  let nesContainers: NESYoRHaHybrid3D[] = [];
  let nesProgressBars: NESYoRHaHybrid3D[] = [];

  onMount(() => {
    initializeScene();
    createDemoComponents();
    startAnimationLoop();
    setupCacheMonitoring();
    
    return () => {
      cleanup();
    };
  });

  function initializeScene(): void {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(NES_YORHA_PALETTE.nesBlack);

    // Camera
    camera = new THREE.PerspectiveCamera(75, 800 / 600, 0.1, 1000);
    camera.position.z = 5;

    // Renderer
    renderer = new THREE.WebGLRenderer({ 
      canvas: canvasElement, 
      antialias: false, // Disable antialiasing for pixelated look
      alpha: true 
    });
    renderer.setSize(800, 600);
    renderer.setPixelRatio(1); // Force 1:1 pixel ratio for retro effect

    // Lighting (minimal for NES aesthetic)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);

    console.log('🎮 3D Scene initialized with NES aesthetic');
  }

  async function createDemoComponents(): Promise<void> {
    isLoading = true;
    
    try {
      // Initialize NES Cache Orchestrator
      await nesCacheOrchestrator.start();

      // Create NES-styled 3D buttons
      await createNESButtons();
      
      // Create NES containers
      await createNESContainers();
      
      // Create progress bars
      await createNESProgressBars();

      console.log('✅ All hybrid components created and cached');
      
    } catch (error) {
      console.error('❌ Error creating demo components:', error);
    } finally {
      isLoading = false;
    }
  }

  async function createNESButtons(): Promise<void> {
    const buttonConfigs = [
      { text: 'Primary', variant: 'is-primary' as const, position: [-3, 2, 0] },
      { text: 'Success', variant: 'is-success' as const, position: [-1, 2, 0] },
      { text: 'Warning', variant: 'is-warning' as const, position: [1, 2, 0] },
      { text: 'Error', variant: 'is-error' as const, position: [3, 2, 0] }
    ];

    for (const config of buttonConfigs) {
      const button = createNESButton({
        text: config.text,
        variant: config.variant,
        size: 'normal',
        onClick: () => handleButtonClick(config.text)
      });

      button.position.set(config.position[0], config.position[1], config.position[2]);
      
      // Add click handler
      button.addEventListener('click', () => {
        console.log(`🎮 NES Button clicked: ${config.text}`);
        playNESClickSound();
      });

      scene.add(button);
      nesButtons.push(button);

      // Cache button states
      await button.cacheCurrentState();
    }
  }

  async function createNESContainers(): Promise<void> {
    const containerConfigs = [
      { title: 'Legal Documents', position: [-2, 0, 0], dark: false },
      { title: 'Evidence Files', position: [2, 0, 0], dark: true },
    ];

    for (const config of containerConfigs) {
      const container = createNESContainer({
        title: config.title,
        dark: config.dark,
        rounded: true
      });

      container.position.set(config.position[0], config.position[1], config.position[2]);
      container.scale.multiplyScalar(0.8);

      scene.add(container);
      nesContainers.push(container);
    }
  }

  async function createNESProgressBars(): Promise<void> {
    const progressConfigs = [
      { value: 75, variant: 'is-success' as const, position: [-1, -2, 0] },
      { value: 45, variant: 'is-warning' as const, position: [1, -2, 0] },
    ];

    for (const config of progressConfigs) {
      const progressBar = createNESProgressBar({
        value: config.value,
        max: 100,
        variant: config.variant
      });

      progressBar.position.set(config.position[0], config.position[1], config.position[2]);

      scene.add(progressBar);
      nesProgressBars.push(progressBar);
    }
  }

  function startAnimationLoop(): void {
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      // Update all hybrid components
      const deltaTime = 0.016; // ~60fps
      
      nesButtons.forEach(button => button.update(deltaTime));
      nesContainers.forEach(container => container.update(deltaTime));
      nesProgressBars.forEach(bar => bar.update(deltaTime));

      // Gentle rotation for demo purposes
      scene.rotation.y += 0.001;

      renderer.render(scene, camera);
    };

    animate();
  }

  function setupCacheMonitoring(): void {
    const updateStats = () => {
      cacheStats = nesCacheOrchestrator.getMemoryStats();
    };

    updateStats();
    setInterval(updateStats, 2000); // Update every 2 seconds
  }

  function handleButtonClick(buttonText: string): void {
    // Simulate progress update
    progress = Math.min(100, progress + 20);
    
    // Switch demo based on button
    switch (buttonText) {
      case 'Primary':
        currentDemo = 'buttons';
        break;
      case 'Success':
        currentDemo = 'containers';
        break;
      case 'Warning':
        currentDemo = 'progress';
        break;
      case 'Error':
        resetDemo();
        break;
    }
  }

  function playNESClickSound(): void {
    // Simulate NES-style click sound (in a real app, use Web Audio API)
    console.log('🔊 *8-bit click sound*');
  }

  async function switchRenderMode(mode: '2d-overlay' | '3d-embedded' | 'hybrid-sync'): Promise<void> {
    for (const button of nesButtons) {
      button.setStyle({ renderMode: mode });
    }
    console.log(`🔄 Switched to render mode: ${mode}`);
  }

  async function toggleCRTEffect(): Promise<void> {
    for (const button of nesButtons) {
      const currentStyle = button.getStyle();
      button.setStyle({ 
        crtEffect: !currentStyle.crtEffect,
        scanlines: !currentStyle.crtEffect 
      });
    }
    console.log('📺 Toggled CRT effect');
  }

  async function clearCache(): Promise<void> {
    await nesCacheOrchestrator.clearRegion('CHR_ROM');
    console.log('🗑️ Cleared sprite cache');
  }

  function resetDemo(): void {
    progress = 0;
    currentDemo = 'buttons';
    
    // Reset all components to default state
    nesButtons.forEach(button => {
      button.switchToNESState(`hybrid_default_${Date.now()}`);
    });
  }

  function cleanup(): void {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }

    // Dispose all components
    [...nesButtons, ...nesContainers, ...nesProgressBars].forEach(component => {
      component.dispose();
    });

    // Dispose renderer
    renderer.dispose();

    console.log('🧹 Demo cleaned up');
  }
</script>

<svelte:head>
  <title>NES.css + YoRHa 3D Hybrid Demo</title>
</svelte:head>

<div class="nes-yorha-3d-demo">
  <!-- Header -->
  <div class="yorha-card p-4 mb-6">
    <h1 class="text-3xl font-bold mb-4 flex items-center gap-4">
      <span class="nes-text is-primary">🎮</span>
      NES.css + YoRHa 3D Hybrid System
    </h1>
    <p class="text-nier-text-secondary mb-4">
      Revolutionary fusion of 8-bit retro styling with advanced 3D GPU interfaces and NES-style memory caching
    </p>

    <div class="flex flex-wrap gap-2 mb-4">
      <span class="nes-badge">
        <span class="is-dark">NES.css</span>
        <span class="is-primary">8-bit DOM</span>
      </span>
      <span class="nes-badge">
        <span class="is-dark">YoRHa 3D</span>
        <span class="is-success">Three.js</span>
      </span>
      <span class="nes-badge">
        <span class="is-dark">WebGL</span>
        <span class="is-warning">GPU Accelerated</span>
      </span>
    </div>

    {#if isLoading}
      <div class="mb-4">
        <progress class="nes-progress is-primary" value={progress} max="100"></progress>
        <p class="nes-text text-center mt-2">Loading hybrid components... {progress}%</p>
      </div>
    {/if}
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-4 gap-6">
    
    <!-- 3D Canvas -->
    <div class="xl:col-span-3">
      <div class="nes-container with-title">
        <p class="title">🎯 3D Hybrid Interface</p>
        
        <div class="canvas-container">
          <canvas bind:this={canvasElement} class="nes-3d-canvas"></canvas>
          
          <!-- DOM Overlay Elements (synchronized with 3D) -->
          <div class="dom-overlay">
            <div class="nes-container is-rounded overlay-info">
              <p class="nes-text is-primary">Current Demo: {currentDemo}</p>
              <p class="nes-text">Progress: {progress}%</p>
            </div>
          </div>
        </div>

        <!-- Controls -->
        <div class="controls mt-4">
          <div class="flex flex-wrap gap-2 mb-2">
            <button 
              class="nes-btn is-primary" 
              click={() => switchRenderMode('hybrid-sync')}
            >
              Hybrid Sync
            </button>
            <button 
              class="nes-btn" 
              click={() => switchRenderMode('2d-overlay')}
            >
              2D Overlay
            </button>
            <button 
              class="nes-btn" 
              click={() => switchRenderMode('3d-embedded')}
            >
              3D Embedded
            </button>
          </div>
          
          <div class="flex flex-wrap gap-2">
            <button 
              class="nes-btn is-warning" 
              click={toggleCRTEffect}
            >
              📺 Toggle CRT
            </button>
            <button 
              class="nes-btn is-error" 
              click={clearCache}
            >
              🗑️ Clear Cache
            </button>
            <button 
              class="nes-btn is-success" 
              click={resetDemo}
            >
              🔄 Reset
            </button>
          </div>

          <!-- Final Fantasy Style Feature Modals -->
          <div class="mt-4 pt-4 border-t border-gray-600">
            <h3 class="nes-text is-primary text-sm mb-2">🎭 Final Fantasy Style Modals</h3>
            <div class="grid grid-cols-2 gap-2">
              <FinalFantasyButton
                variant="magic"
                size="small"
                click={() => showYoRHaModal = true}
                icon="🤖"
              >
                {#snippet children()}YoRHa 3D{/snippet}
              </FinalFantasyButton>

              <FinalFantasyButton
                variant="item"
                size="small" 
                click={() => showNESModal = true}
                icon="🎮"
              >
                {#snippet children()}NES Cache{/snippet}
              </FinalFantasyButton>

              <FinalFantasyButton
                variant="success"
                size="small"
                click={() => showWebGLModal = true}
                icon="🌟"
              >
                {#snippet children()}WebGL GPU{/snippet}
              </FinalFantasyButton>

              <FinalFantasyButton
                variant="primary"
                size="small"
                click={() => showHybridModal = true}
                icon="⚡"
              >
                {#snippet children()}Hybrid Sync{/snippet}
              </FinalFantasyButton>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Control Panel -->
    <div class="xl:col-span-1 space-y-6">
      
      <!-- Cache Stats -->
      <div class="nes-container with-title is-rounded">
        <p class="title">💾 NES Memory Stats</p>
        
        {#if cacheStats}
          <div class="space-y-2 text-xs">
            <div>
              <strong>Total Usage:</strong>
              <progress 
                class="nes-progress is-pattern" 
                value={cacheStats.utilization * 100} 
                max="100"
              ></progress>
              <span>{Math.round(cacheStats.utilization * 100)}%</span>
            </div>
            
            <div class="mt-3">
              <strong>Memory Regions:</strong>
              <ul class="nes-list">
                <li>PRG_ROM: {cacheStats.regions?.PRG_ROM || '0/32KB'}</li>
                <li>CHR_ROM: {cacheStats.regions?.CHR_ROM || '0/8KB'}</li>
                <li>RAM: {cacheStats.regions?.RAM || '0/2KB'}</li>
                <li>PPU: {cacheStats.regions?.PPU_MEMORY || '0/16KB'}</li>
              </ul>
            </div>

            <div class="mt-3">
              <strong>Cache Entries:</strong>
              <ul class="nes-list">
                <li>Sprites: {cacheStats.caches?.spritesheets || 0}</li>
                <li>YoRHa UI: {cacheStats.caches?.yorhaComponents || 0}</li>
                <li>Animations: {cacheStats.caches?.animations || 0}</li>
                <li>Shaders: {cacheStats.caches?.webgpuShaders || 0}</li>
              </ul>
            </div>
          </div>
        {:else}
          <p class="nes-text">Loading cache statistics...</p>
        {/if}
      </div>

      <!-- Demo Controls -->
      <div class="nes-container with-title is-rounded">
        <p class="title">🎮 Demo Controls</p>
        
        <div class="space-y-2">
          <button 
            class="nes-btn w-full"
            class:is-primary={currentDemo === 'buttons'}
            click={() => currentDemo = 'buttons'}
          >
            🔘 NES Buttons
          </button>
          
          <button 
            class="nes-btn w-full"
            class:is-primary={currentDemo === 'containers'}
            click={() => currentDemo = 'containers'}
          >
            📦 Containers
          </button>
          
          <button 
            class="nes-btn w-full"
            class:is-primary={currentDemo === 'progress'}
            click={() => currentDemo = 'progress'}
          >
            📊 Progress Bars
          </button>
        </div>

        <div class="mt-4">
          <div class="nes-field">
            <label>Animation Speed</label>
            <input type="range" class="w-full" min="1" max="100" value="50">
          </div>
        </div>
      </div>

      <!-- Feature List -->
      <div class="nes-container with-title is-dark">
        <p class="title">⚡ Hybrid Features</p>
        
        <div class="nes-list text-xs">
          <ul>
            <li>✅ NES.css DOM Styling</li>
            <li>✅ YoRHa 3D Components</li>
            <li>✅ WebGL GPU Acceleration</li>
            <li>✅ NES-Style Caching</li>
            <li>✅ CRT/Scanline Effects</li>
            <li>✅ Hybrid Sync Rendering</li>
            <li>✅ Pixel-Perfect Geometry</li>
            <li>✅ 8-bit Color Quantization</li>
            <li>✅ Predictive State Loading</li>
            <li>✅ Memory Budget Management</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div class="mt-6 nes-container is-rounded">
    <p class="nes-text text-center">
      🎮 <strong>Revolutionary Interface:</strong> NES-era efficiency meets modern 3D graphics
    </p>
    <p class="nes-text text-center text-xs mt-2">
      Combining 8-bit DOM aesthetics with GPU-accelerated 3D rendering and intelligent caching
    </p>
  </div>

  <!-- Final Fantasy Style Modals -->
  
  <!-- YoRHa 3D Components Modal -->
  <FinalFantasyModal
    bind:isOpen={showYoRHaModal}
    title="YoRHa 3D Components System"
    type="battle"
    size="large"
    cornerStyle="hybrid"
    close={() => showYoRHaModal = false}
  >
    {#snippet children()}
      <FinalFantasyContainer title="Advanced Three.js-powered 3D Interface Elements" type="battle" glowEffect={true}>
        {#snippet children()}
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div>
                <h4 class="text-yellow-300 font-bold mb-2">🎯 Core Features:</h4>
                <ul class="text-white text-sm space-y-1">
                  <li>✓ CSS-like styling for Three.js objects</li>
                  <li>✓ Material creation and management</li>
                  <li>✓ Advanced animation systems</li>
                  <li>✓ Interactive 3D components</li>
                  <li>✓ Memory-efficient rendering</li>
                  <li>✓ GPU acceleration support</li>
                </ul>
              </div>
              <div>
                <h4 class="text-yellow-300 font-bold mb-2">🔧 Technical Stack:</h4>
                <ul class="text-white text-sm space-y-1">
                  <li>• Three.js WebGL renderer</li>
                  <li>• Custom shader materials</li>
                  <li>• Scene graph management</li>
                  <li>• Event handling system</li>
                  <li>• Component lifecycle hooks</li>
                  <li>• Texture streaming pipeline</li>
                </ul>
              </div>
            </div>
            
            <div class="mt-4 p-4 bg-black/30 rounded border border-yellow-500/30">
              <h4 class="text-cyan-300 font-bold mb-2">🎮 Integration Points:</h4>
              <p class="text-white text-sm">
                YoRHa 3D components seamlessly integrate with DOM elements through synchronized transforms, 
                shared state management, and unified event handling. This creates a hybrid interface where 
                3D objects can interact with traditional web UI elements.
              </p>
            </div>
          </div>
        {/snippet}
      </FinalFantasyContainer>
    {/snippet}

    {#snippet actions()}
      <FinalFantasyButton variant="secondary" click={() => showYoRHaModal = false}>
        {#snippet children()}Close{/snippet}
      </FinalFantasyButton>
    {/snippet}
  </FinalFantasyModal>

  <!-- NES-Style Caching Modal -->
  <FinalFantasyModal
    bind:isOpen={showNESModal}
    title="NES-Style Memory Architecture"
    type="inventory"
    size="large"
    cornerStyle="classic"
    close={() => showNESModal = false}
  >
    {#snippet children()}
      <FinalFantasyContainer title="Memory-efficient State Management" type="inventory" glowEffect={true}>
        {#snippet children()}
          <div class="space-y-4">
            <div class="grid grid-cols-3 gap-4">
              <div class="bg-amber-900/20 p-3 rounded border border-amber-500/30">
                <h4 class="text-amber-300 font-bold text-sm">PRG_ROM (32KB)</h4>
                <p class="text-white text-xs mt-1">Program logic cache</p>
                <div class="mt-2 bg-amber-600 h-2 rounded-full"></div>
                <span class="text-amber-200 text-xs">85% utilized</span>
              </div>
              <div class="bg-green-900/20 p-3 rounded border border-green-500/30">
                <h4 class="text-green-300 font-bold text-sm">CHR_ROM (8KB)</h4>
                <p class="text-white text-xs mt-1">Pattern/texture data</p>
                <div class="mt-2 bg-green-600 h-2 rounded-full w-3/4"></div>
                <span class="text-green-200 text-xs">72% utilized</span>
              </div>
              <div class="bg-blue-900/20 p-3 rounded border border-blue-500/30">
                <h4 class="text-blue-300 font-bold text-sm">RAM (2KB)</h4>
                <p class="text-white text-xs mt-1">Working memory</p>
                <div class="mt-2 bg-blue-600 h-2 rounded-full w-1/2"></div>
                <span class="text-blue-200 text-xs">45% utilized</span>
              </div>
            </div>

            <div class="p-4 bg-black/40 rounded border border-cyan-500/30">
              <h4 class="text-cyan-300 font-bold mb-2">🧠 Intelligent Caching Features:</h4>
              <div class="grid grid-cols-2 gap-4 text-sm">
                <ul class="text-white space-y-1">
                  <li>• Predictive state preloading</li>
                  <li>• LRU eviction policies</li>
                  <li>• Memory budget enforcement</li>
                  <li>• Cache warming strategies</li>
                </ul>
                <ul class="text-white space-y-1">
                  <li>• Background cleanup tasks</li>
                  <li>• Usage pattern analysis</li>
                  <li>• Performance monitoring</li>
                  <li>• Memory fragmentation handling</li>
                </ul>
              </div>
            </div>
          </div>
        {/snippet}
      </FinalFantasyContainer>
    {/snippet}

    {#snippet actions()}
      <FinalFantasyButton variant="secondary" click={() => showNESModal = false}>
        {#snippet children()}Close{/snippet}
      </FinalFantasyButton>
    {/snippet}
  </FinalFantasyModal>

  <!-- WebGL GPU Modal -->
  <FinalFantasyModal
    bind:isOpen={showWebGLModal}
    title="WebGL GPU Acceleration System"
    type="magic"
    size="large"
    cornerStyle="modern"
    close={() => showWebGLModal = false}
  >
    <FinalFantasyContainer title="High-Performance Graphics Processing" type="magic" glowEffect={true}>
      <div class="space-y-4">
        <div class="grid grid-cols-2 gap-6">
          <div>
            <h4 class="text-purple-300 font-bold mb-3">🌟 GPU Features:</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-white">Shader Compilation:</span>
                <span class="text-green-300">✓ Active</span>
              </div>
              <div class="flex justify-between">
                <span class="text-white">Texture Streaming:</span>
                <span class="text-green-300">✓ Optimized</span>
              </div>
              <div class="flex justify-between">
                <span class="text-white">Vertex Buffer Objects:</span>
                <span class="text-green-300">✓ Cached</span>
              </div>
              <div class="flex justify-between">
                <span class="text-white">Instanced Rendering:</span>
                <span class="text-green-300">✓ Enabled</span>
              </div>
            </div>
          </div>
          <div>
            <h4 class="text-purple-300 font-bold mb-3">⚡ Performance Stats:</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-white">Frame Rate:</span>
                <span class="text-yellow-300">60 FPS</span>
              </div>
              <div class="flex justify-between">
                <span class="text-white">GPU Memory:</span>
                <span class="text-yellow-300">2.1GB / 8GB</span>
              </div>
              <div class="flex justify-between">
                <span class="text-white">Draw Calls:</span>
                <span class="text-yellow-300">127/frame</span>
              </div>
              <div class="flex justify-between">
                <span class="text-white">Triangles:</span>
                <span class="text-yellow-300">45.2K</span>
              </div>
            </div>
          </div>
        </div>

        <div class="p-4 bg-purple-900/20 rounded border border-purple-500/30">
          <h4 class="text-purple-300 font-bold mb-2">🎨 Rendering Pipeline:</h4>
          <p class="text-white text-sm">
            The WebGL rendering pipeline processes 3D geometry through vertex and fragment shaders, 
            applies materials and lighting, and outputs to frame buffers. Advanced features include 
            shadow mapping, post-processing effects, and multi-pass rendering for complex visual effects.
          </p>
        </div>
      </div>
    </FinalFantasyContainer>

    <svelte:fragment slot="actions">
      <FinalFantasyButton variant="secondary" click={() => showWebGLModal = false}>
        Close
      </FinalFantasyButton>
    </svelte:fragment>
  </FinalFantasyModal>

  <!-- Hybrid Rendering Modes Modal -->
  <FinalFantasyModal
    bind:isOpen={showHybridModal}
    title="Hybrid Rendering Synchronization"
    type="menu"
    size="large"
    cornerStyle="hybrid"
    close={() => showHybridModal = false}
  >
    <FinalFantasyContainer title="Seamless DOM-WebGL Integration" type="menu" glowEffect={true}>
      <div class="space-y-4">
        <div class="grid grid-cols-3 gap-4">
          <div class="text-center p-3 bg-blue-900/20 rounded border border-blue-500/30">
            <h4 class="text-blue-300 font-bold text-sm mb-2">🎭 2D Overlay</h4>
            <p class="text-white text-xs">DOM elements positioned over 3D scene</p>
            <div class="mt-2 w-full h-8 bg-gradient-to-r from-blue-600 to-blue-400 rounded opacity-75"></div>
          </div>
          <div class="text-center p-3 bg-green-900/20 rounded border border-green-500/30">
            <h4 class="text-green-300 font-bold text-sm mb-2">🎮 3D Embedded</h4>
            <p class="text-white text-xs">3D objects rendered as DOM textures</p>
            <div class="mt-2 w-full h-8 bg-gradient-to-r from-green-600 to-green-400 rounded opacity-75"></div>
          </div>
          <div class="text-center p-3 bg-purple-900/20 rounded border border-purple-500/30">
            <h4 class="text-purple-300 font-bold text-sm mb-2">⚡ Hybrid Sync</h4>
            <p class="text-white text-xs">Real-time DOM-WebGL synchronization</p>
            <div class="mt-2 w-full h-8 bg-gradient-to-r from-purple-600 to-purple-400 rounded opacity-75"></div>
          </div>
        </div>

        <div class="p-4 bg-black/40 rounded border border-cyan-500/30">
          <h4 class="text-cyan-300 font-bold mb-2">🔄 Synchronization Features:</h4>
          <div class="grid grid-cols-2 gap-4 text-sm">
            <ul class="text-white space-y-1">
              <li>✓ Transform matrix sharing</li>
              <li>✓ Event handler bridging</li>
              <li>✓ State synchronization</li>
              <li>✓ Animation coordination</li>
            </ul>
            <ul class="text-white space-y-1">
              <li>✓ Collision detection</li>
              <li>✓ Z-index management</li>
              <li>✓ Viewport adaptation</li>
              <li>✓ Performance optimization</li>
            </ul>
          </div>
        </div>

        <div class="p-4 bg-gradient-to-r from-yellow-900/20 to-orange-900/20 rounded border border-yellow-500/30">
          <h4 class="text-yellow-300 font-bold mb-2">🎯 Revolutionary Achievement:</h4>
          <p class="text-white text-sm">
            This hybrid system bridges the gap between traditional DOM manipulation and modern 3D graphics, 
            enabling developers to leverage the best of both worlds while maintaining optimal performance 
            and visual fidelity.
          </p>
        </div>
      </div>
    </FinalFantasyContainer>

    <svelte:fragment slot="actions">
      <FinalFantasyButton variant="secondary" click={() => showHybridModal = false}>
        Close
      </FinalFantasyButton>
    </svelte:fragment>
  </FinalFantasyModal>
</div>

<style>
  .nes-yorha-3d-demo {
    min-height: 100vh;
    padding: 2rem;
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0f0f0f 100%);
  }

  .canvas-container {
    position: relative;
    width: 100%;
    background: #000;
    border: 4px solid #222;
    border-radius: 8px;
    overflow: hidden;
  }

  .nes-3d-canvas {
    display: block;
    width: 100%;
    height: 600px;
    image-rendering: pixelated; /* Preserve pixel-perfect rendering */
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }

  .dom-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    pointer-events: none;
    z-index: 10;
  }

  .overlay-info {
    background: rgba(26, 26, 26, 0.9);
    border-color: #ffd700;
    pointer-events: auto;
  }

  .controls {
    background: #1a1a1a;
    border: 2px solid #333;
    border-radius: 4px;
    padding: 1rem;
  }

  /* NES-style animations */
  .nes-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);
  }

  .nes-btn:active {
    transform: translateY(0);
  }

  /* Progress bar enhancements */
  .nes-progress {
    background: #222;
  }

  /* Badge styling */
  .nes-badge {
    display: inline-block;
    font-size: 0.75rem;
  }

  /* Responsive adjustments */
  @media (max-width: 1024px) {
    .nes-yorha-3d-demo {
      padding: 1rem;
    }
    
    .nes-3d-canvas {
      height: 400px;
    }
  }

  @media (max-width: 640px) {
    .grid-cols-1.xl\\:grid-cols-4 {
      gap: 1rem;
    }
    
    .nes-3d-canvas {
      height: 300px;
    }

    .controls .flex {
      flex-direction: column;
    }
  }
</style>