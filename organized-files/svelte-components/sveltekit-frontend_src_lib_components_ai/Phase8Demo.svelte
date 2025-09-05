<script lang="ts">
  import { onMount } from 'svelte';
  import { MatrixUICompiler, type MatrixUINode } from '$lib/ui/matrix-compiler';
  import { MatrixLODSystem, type ViewportFocus } from '$lib/ui/matrix-lod';
  import { LegalAIReranker, type UserContext, enhancedSearch } from '$lib/ai/custom-reranker';
  import { PredictivePrefetcher } from '$lib/workers/predictive-prefetch';

  interface Props {
    className?: string;
  }

  let { className = '' } = $props();
  
  // Phase 8 system components
  let matrixCompiler: MatrixUICompiler
  let lodSystem: MatrixLODSystem
  let reranker: LegalAIReranker
  let prefetcher: PredictivePrefetcher
  
  // Demo state
  let canvas: HTMLCanvasElement
  let demoContainer: HTMLDivElement
  let isSystemInitialized = $state(false);
  let currentDemo = $state<'reranker' | 'matrix' | 'lod' | 'prefetch'>('reranker');
  let performanceMetrics = $state({
    frameRate: 0,
    lodLevels: { low: 0, mid: 0, high: 0 },
    cacheHits: 0,
    aiBoosts: 0
  });
  
  // Demo data
  const sampleUIDefinition: MatrixUINode[] = [
    {
      type: 'card',
      id: 'evidence-card-1',
      matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 50, 0, 1],
      styles: {
        base: 'yorha-card p-6 bg-gray-900 border border-yellow-400'
      },
      events: ['click', 'mouseover'],
      metadata: {
        priority: 'high',
        confidence: 95,
        evidenceType: 'forensic',
        aiGenerated: true
      }
    },
    {
      type: 'button',
      id: 'analyze-btn-1',
      matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 250, 150, 0, 1],
      styles: {
        base: 'yorha-button px-4 py-2 bg-yellow-400 text-black'
      },
      events: ['click'],
      metadata: {
        priority: 'critical',
        confidence: 88,
        aiGenerated: false
      }
    },
    {
      type: 'evidence-item',
      id: 'evidence-item-1',
      matrix: [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 1, 0, 400, 100, 0, 1],
      styles: {
        base: 'yorha-evidence-item border-l-4 border-blue-400 pl-4'
      },
      events: ['click', 'dblclick'],
      metadata: {
        priority: 'medium',
        confidence: 72,
        evidenceType: 'digital',
        aiGenerated: true
      }
    }
  ];
  
  const sampleUserContext: UserContext = {
    intent: 'analyze',
    timeOfDay: 'afternoon',
    focusedElement: 'evidence-card-1',
    currentCase: 'CASE-2024-001',
    recentActions: ['file_upload', 'view_document', 'apply_filter'],
    userRole: 'prosecutor',
    workflowState: 'review'
  };

  onMount(async () => {
    await initializePhase8System();
    startDemoLoop();
  });

  async function initializePhase8System(): Promise<void> {
    try {
      // Initialize Matrix UI Compiler
      matrixCompiler = new MatrixUICompiler(canvas);
      
      // Initialize LOD System
      lodSystem = new MatrixLODSystem(canvas);
      
      // Initialize AI Reranker
      reranker = new LegalAIReranker();
      
      // Initialize Predictive Prefetcher
      prefetcher = new PredictivePrefetcher();
      await prefetcher.initialize();
      
      // Compile sample UI
      const compiledNodes = await matrixCompiler.compile(sampleUIDefinition);
      
      // Build LOD cache for each component
      compiledNodes.forEach(node => {
        const sourceNode = sampleUIDefinition.find(n => n.id === node.element.id);
        if (sourceNode) {
          const vertices = new Float32Array([
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0
          ]);
          
          lodSystem.buildLODCache(sourceNode.id, vertices, sourceNode.metadata || {});
        }
      });
      
      // Render components to demo container
      compiledNodes.forEach(node => {
        demoContainer.appendChild(node.element);
      });
      
      isSystemInitialized = true;
      console.log('✅ Phase 8 AI-Aware Matrix UI System initialized successfully');
      
    } catch (error) {
      console.error('❌ Phase 8 system initialization failed:', error);
    }
  }

  function startDemoLoop(): void {
    let frameCount = 0;
    
    const demoLoop = () => {
      frameCount++;
      
      // Update performance metrics
      const metrics = lodSystem.getPerformanceMetrics();
      performanceMetrics.frameRate = metrics.frameRate;
      
      // Simulate viewport focus changes
      if (frameCount % 120 === 0) { // Every 2 seconds at 60fps
        simulateViewportFocus();
      }
      
      // Simulate AI suggestions
      if (frameCount % 180 === 0) { // Every 3 seconds
        simulateAISuggestions();
      }
      
      requestAnimationFrame(demoLoop);
    };
    
    requestAnimationFrame(demoLoop);
  }

  function simulateViewportFocus(): void {
    const focus: ViewportFocus = {
      centerX: Math.random() * 800,
      centerY: Math.random() * 600,
      radius: 200 + Math.random() * 100,
      aiSuggestions: ['evidence-card-1', 'analyze-btn-1'],
      confidenceScore: 0.8 + Math.random() * 0.2
    };
    
    lodSystem.updateViewportFocus(focus);
  }

  async function simulateAISuggestions(): Promise<void> {
    try {
      // Simulate AI reranking
      const mockResults = [
        {
          id: 'evidence-1',
          content: 'Forensic DNA analysis report',
          metadata: { type: 'evidence-analysis', confidence: 95 },
          originalScore: 0.8,
          rerankScore: 0,
          confidence: 95
        },
        {
          id: 'precedent-1',
          content: 'Similar case precedent from 2023',
          metadata: { type: 'case-precedent', confidence: 78 },
          originalScore: 0.6,
          rerankScore: 0,
          confidence: 78
        }
      ];
      
      const rerankedResults = await reranker.rerank(mockResults, sampleUserContext);
      performanceMetrics.aiBoosts = rerankedResults.filter(r => r.rerankScore > r.originalScore).length;
      
      // Simulate predictive prefetching
      const intentPrediction = await prefetcher.predictIntent(sampleUserContext.context);
      if (intentPrediction) {
        await prefetcher.executePrefetch(intentPrediction);
        performanceMetrics.cacheHits++;
      }
      
    } catch (error) {
      console.warn('Demo simulation error:', error);
    }
  }

  function switchDemo(demo: typeof currentDemo): void {
    currentDemo = demo;
    
    // Reset visual indicators
    document.querySelectorAll('[id^="evidence-"], [id^="analyze-"]').forEach(el => {
      el.classList.remove('demo-highlight', 'ai-enhanced', 'lod-demo');
    });
    
    // Apply demo-specific styling
    switch (demo) {
      case 'reranker':
        document.getElementById('evidence-card-1')?.classList.add('demo-highlight');
        break;
      case 'matrix':
        document.getElementById('analyze-btn-1')?.classList.add('ai-enhanced');
        break;
      case 'lod':
        document.querySelectorAll('[id^="evidence-"]').forEach(el => {
          el.classList.add('lod-demo');
        });
        break;
      case 'prefetch':
        // Show prefetch indicators
        break;
    }
  }

  function runPerformanceTest(): void {
    console.log('🚀 Running Phase 8 Performance Test...');
    
    // Stress test the LOD system
    for (let i = 0; i < 100; i++) {
      setTimeout(() => {
        simulateViewportFocus();
      }, i * 10);
    }
    
    // Stress test the reranker
    for (let i = 0; i < 50; i++) {
      setTimeout(() => {
        simulateAISuggestions();
      }, i * 20);
    }
    
    setTimeout(() => {
      console.log('✅ Performance test completed');
      console.log('📊 Metrics:', performanceMetrics);
    }, 2000);
  }
</script>

<div class="phase8-demo {className}">
  <div class="demo-header yorha-panel p-6 mb-6">
    <h2 class="text-2xl font-bold text-yellow-400 mb-2">
      Phase 8: AI-Aware Matrix UI System
    </h2>
    <p class="text-gray-300 mb-4">
      Complete integration of Custom Reranker + JSON UI Compiler + WebGL LOD + Predictive Prefetching
    </p>
    
    <!-- System Status -->
    <div class="system-status flex items-center gap-4 mb-4">
      <div class="status-item">
        <span class="indicator {isSystemInitialized ? 'bg-green-400' : 'bg-red-400'}"></span>
        <span class="text-sm">System {isSystemInitialized ? 'Online' : 'Offline'}</span>
      </div>
      <div class="status-item">
        <span class="text-sm">FPS: {performanceMetrics.frameRate}</span>
      </div>
      <div class="status-item">
        <span class="text-sm">AI Boosts: {performanceMetrics.aiBoosts}</span>
      </div>
      <div class="status-item">
        <span class="text-sm">Cache Hits: {performanceMetrics.cacheHits}</span>
      </div>
    </div>
    
    <!-- Demo Controls -->
    <div class="demo-controls flex gap-2 mb-4">
      <button 
        class="yorha-button px-3 py-1 text-sm {currentDemo === 'reranker' ? 'bg-yellow-400 text-black' : 'bg-gray-700'}"
        onclick={() => switchDemo('reranker')}
      >
        AI Reranker
      </button>
      <button 
        class="yorha-button px-3 py-1 text-sm {currentDemo === 'matrix' ? 'bg-yellow-400 text-black' : 'bg-gray-700'}"
        onclick={() => switchDemo('matrix')}
      >
        Matrix Compiler
      </button>
      <button 
        class="yorha-button px-3 py-1 text-sm {currentDemo === 'lod' ? 'bg-yellow-400 text-black' : 'bg-gray-700'}"
        onclick={() => switchDemo('lod')}
      >
        LOD System
      </button>
      <button 
        class="yorha-button px-3 py-1 text-sm {currentDemo === 'prefetch' ? 'bg-yellow-400 text-black' : 'bg-gray-700'}"
        onclick={() => switchDemo('prefetch')}
      >
        Prefetcher
      </button>
      <button 
        class="yorha-button px-3 py-1 text-sm bg-blue-600 text-white ml-4"
        onclick={runPerformanceTest}
      >
        Performance Test
      </button>
    </div>
  </div>

  <!-- Demo Content -->
  <div class="demo-content grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- WebGL Canvas -->
    <div class="webgl-section">
      <h3 class="text-lg font-semibold text-yellow-400 mb-3">WebGL2 Rendering</h3>
      <div class="canvas-container relative bg-black border border-gray-700 rounded">
        <canvas 
          bind:this={canvas}
          width="600"
          height="400"
          class="w-full h-auto"
        ></canvas>
        <div class="overlay absolute top-2 left-2 text-xs text-green-400 font-mono">
          LOD System Active | Cubic Filter Blending | AI-Aware Rendering
        </div>
      </div>
    </div>
    
    <!-- UI Components Demo -->
    <div class="ui-section">
      <h3 class="text-lg font-semibold text-yellow-400 mb-3">Matrix UI Components</h3>
      <div 
        bind:this={demoContainer}
        class="ui-container relative h-96 bg-gray-900 border border-gray-700 rounded p-4 overflow-hidden"
      >
        <!-- Components will be rendered here by the matrix compiler -->
      </div>
    </div>
  </div>

  <!-- Technical Details -->
  <div class="technical-details mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    <div class="tech-card yorha-panel p-4">
      <h4 class="text-sm font-semibold text-yellow-400 mb-2">Custom Reranker</h4>
      <div class="text-xs text-gray-300 space-y-1">
        <div>✅ Legal context awareness</div>
        <div>✅ Role-based scoring</div>
        <div>✅ Workflow state logic</div>
        <div>✅ Confidence weighting</div>
      </div>
    </div>
    
    <div class="tech-card yorha-panel p-4">
      <h4 class="text-sm font-semibold text-yellow-400 mb-2">Matrix Compiler</h4>
      <div class="text-xs text-gray-300 space-y-1">
        <div>✅ JSON to DOM conversion</div>
        <div>✅ 4x4 matrix transforms</div>
        <div>✅ UnoCSS integration</div>
        <div>✅ Event handling</div>
      </div>
    </div>
    
    <div class="tech-card yorha-panel p-4">
      <h4 class="text-sm font-semibold text-yellow-400 mb-2">LOD System</h4>
      <div class="text-xs text-gray-300 space-y-1">
        <div>✅ GLSL cubic blending</div>
        <div>✅ AI priority boosting</div>
        <div>✅ Performance adaptation</div>
        <div>✅ GPU load balancing</div>
      </div>
    </div>
    
    <div class="tech-card yorha-panel p-4">
      <h4 class="text-sm font-semibold text-yellow-400 mb-2">Predictive Prefetch</h4>
      <div class="text-xs text-gray-300 space-y-1">
        <div>✅ Intent prediction</div>
        <div>✅ Service Worker caching</div>
        <div>✅ Behavioral analysis</div>
        <div>✅ Resource optimization</div>
      </div>
    </div>
  </div>

  <!-- Current Demo Info -->
  <div class="demo-info mt-6 yorha-panel p-4">
    {#if currentDemo === 'reranker'}
      <h4 class="text-lg font-semibold text-yellow-400 mb-2">AI Reranker Demo</h4>
      <p class="text-gray-300 text-sm">
        Watch as the AI reranker intelligently scores legal documents based on user context, 
        role, and workflow state. The highlighted evidence card receives priority scoring 
        for prosecutor workflows.
      </p>
    {:else if currentDemo === 'matrix'}
      <h4 class="text-lg font-semibold text-yellow-400 mb-2">Matrix Compiler Demo</h4>
      <p class="text-gray-300 text-sm">
        The JSON UI compiler converts structured definitions into DOM elements with 
        4x4 matrix transforms. AI-enhanced components show subtle glow effects and 
        confidence-based styling.
      </p>
    {:else if currentDemo === 'lod'}
      <h4 class="text-lg font-semibold text-yellow-400 mb-2">LOD System Demo</h4>
      <p class="text-gray-300 text-sm">
        Real-time Level of Detail system with cubic filter blending. Components 
        automatically adjust quality based on viewport focus, AI suggestions, 
        and GPU performance.
      </p>
    {:else if currentDemo === 'prefetch'}
      <h4 class="text-lg font-semibold text-yellow-400 mb-2">Predictive Prefetch Demo</h4>
      <p class="text-gray-300 text-sm">
        Service Worker analyzes user behavior patterns to predict next actions 
        and preload resources. Legal workflow awareness enables intelligent 
        caching strategies.
      </p>
    {/if}
  </div>
</div>

<style>
  /* @unocss-include */
  .phase8-demo {
    @apply max-w-7xl mx-auto p-6;
  }

  .indicator {
    @apply w-2 h-2 rounded-full inline-block mr-2;
  }

  .status-item {
    @apply flex items-center text-sm text-gray-300;
  }

  .tech-card {
    @apply bg-gray-800 border border-gray-700;
  }

  /* Demo-specific styling */
  :global(.demo-highlight) {
    @apply ring-2 ring-yellow-400 ring-opacity-50 bg-yellow-400 bg-opacity-10;
    animation: pulse 2s ease-in-out infinite;
  }

  :global(.ai-enhanced) {
    @apply ring-2 ring-blue-400 ring-opacity-50;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
  }

  :global(.lod-demo) {
    transition: transform 0.3s ease, opacity 0.3s ease;
  }

  :global(.lod-low) {
    @apply opacity-60 scale-95;
  }

  :global(.lod-mid) {
    @apply opacity-80 scale-95;
  }

  :global(.lod-high) {
    @apply opacity-100 scale-100;
  }

  .canvas-container canvas {
    background: linear-gradient(45deg, #1a1a1a 0%, #2d2d2d 100%);
  }

  .overlay {
    background: rgba(0, 0, 0, 0.7);
    padding: 4px 8px;
    border-radius: 4px;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
</style>