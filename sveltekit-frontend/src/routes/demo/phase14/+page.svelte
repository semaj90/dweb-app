<!--
  Phase 14 Complete Demo System - Performance Monitoring
  
  Comprehensive demonstration of all Phase 14 components:
  - Unified UI Kit with bits-ui v2 + Melt Svelte 5
  - WebGPU texture streaming with NES memory constraints
  - GPU-accelerated animations with WebGL shaders
  - NES memory architecture with Nintendo-inspired regions
  - Loki.js + Redis high-performance caching
  - Legal-BERT semantic analysis with real-time AI
  - Performance monitoring and metrics dashboard
-->

<script lang=\"ts\">
  import { onMount, onDestroy } from 'svelte';
  import { fade, fly, scale } from 'svelte/transition';
  import { UnifiedButton, UnifiedDialog, type UnifiedButtonProps } from '$lib/components/unified/index.js';
  import { textureStreamer } from '$lib/webgpu/texture-streaming.js';
  import { gpuAnimations } from '$lib/animations/gpu-animations.js';
  import { nesMemory, type MemoryStats } from '$lib/memory/nes-memory-architecture.js';
  import { lokiRedisCache, type CacheStats } from '$lib/cache/loki-redis-integration.js';
  import { legalBERTAnalyzer, type SemanticAnalysis } from '$lib/ai/legal-bert-semantic-analyzer.js';
  
  // Component state
  let isInitialized = $state(false);
  let currentDemo = $state('overview');
  let performanceMetrics = $state<{
    texture: any;
    memory: MemoryStats | null;
    cache: CacheStats | null;
    ai: any;
    gpu: any;
  }>({
    texture: null,
    memory: null,
    cache: null,
    ai: null,
    gpu: null
  });
  
  // Demo content state
  let sampleDocument = $state('');
  let analysisResult = $state<SemanticAnalysis | null>(null);
  let showDialog = $state(false);
  let animationIntensity = $state(0.5);
  let memoryPressure = $state(false);
  
  // Performance monitoring
  let metricsInterval: NodeJS.Timeout;
  let frameCount = $state(0);
  let fps = $state(0);
let lastFrameTime = $state(0);
  
  // Demo scenarios
  const demoScenarios = [
    {
      id: 'overview',
      title: 'System Overview',
      description: 'Complete Phase 14 architecture overview with real-time metrics'
    },
    {
      id: 'ui_components',
      title: 'Unified UI Components',
      description: 'bits-ui v2 + Melt Svelte 5 + UnoCSS integration showcase'
    },
    {
      id: 'webgpu_streaming',
      title: 'WebGPU Texture Streaming',
      description: 'NES-inspired memory constraints with GPU acceleration'
    },
    {
      id: 'animations',
      title: 'GPU Animations',
      description: 'WebGL shader-based legal confidence animations'
    },
    {
      id: 'memory_architecture',
      title: 'NES Memory Architecture',
      description: 'Nintendo-inspired memory regions with legal document storage'
    },
    {
      id: 'caching_system',
      title: 'Loki.js + Redis Cache',
      description: 'High-performance hybrid caching with real-time sync'
    },
    {
      id: 'ai_analysis',
      title: 'Legal-BERT Analysis',
      description: 'Real-time semantic analysis with transformer models'
    },
    {
      id: 'performance',
      title: 'Performance Dashboard',
      description: 'Comprehensive system performance monitoring'
    }
  ];
  
  // Sample legal documents for testing
  const sampleDocuments = [
    {
      title: 'Software License Agreement',
      content: `This Software License Agreement (\"Agreement\") is entered into between TechCorp Inc. (\"Licensor\") and the end user (\"Licensee\"). The Licensee agrees to indemnify and hold harmless the Licensor from any claims arising from the use of this software. This Agreement shall terminate immediately upon breach of any terms herein. The software is provided \"as is\" without warranty of any kind, express or implied.`
    },
    {
      title: 'Employment Contract',
      content: `This Employment Agreement is between DataSystems LLC (\"Company\") and John Smith (\"Employee\"). Employee warrants that they will maintain confidentiality of all proprietary information. The Company may terminate this agreement at will with 30 days notice. Employee shall not compete with Company for a period of 12 months following termination.`
    },
    {
      title: 'Real Estate Purchase Agreement',
      content: `Purchase Agreement for property located at 123 Main Street. Buyer agrees to purchase property for $500,000. Closing date shall be no later than December 31, 2024. Seller warrants clear title and agrees to provide title insurance. If buyer defaults, earnest money shall be forfeited as liquidated damages.`
    }
  ];

  onMount(async () => {
    try {
      console.log('🚀 Initializing Phase 14 Demo System...');
      
      // Initialize all systems
      await Promise.all([
        textureStreamer.initialize(),
        gpuAnimations.initialize(),
        lokiRedisCache.initialize(),
        legalBERTAnalyzer.initialize()
      ]);
      
      // Start performance monitoring
      startPerformanceMonitoring();
      
      // Load sample document
      sampleDocument = sampleDocuments[0].content;
      
      isInitialized = true;
      console.log('✅ Phase 14 Demo System initialized successfully');
      
    } catch (error) {
      console.error('❌ Demo system initialization failed:', error);
    }
  });

  onDestroy(() => {
    if (metricsInterval) {
      clearInterval(metricsInterval);
    }
  });

  function startPerformanceMonitoring() {
    // Update metrics every second
    metricsInterval = setInterval(async () => {
      try {
        // Update performance metrics
        performanceMetrics = {
          texture: textureStreamer.getMemoryStats(),
          memory: nesMemory.getMemoryStats(),
          cache: lokiRedisCache.getStats(),
          ai: legalBERTAnalyzer.getStats(),
          gpu: gpuAnimations.getPerformanceStats()
        };
        
        // Update FPS
        const currentTime = Date.now();
        if (lastFrameTime > 0) {
          const deltaTime = currentTime - lastFrameTime;
          fps = Math.round(1000 / deltaTime);
        }
        lastFrameTime = currentTime;
        frameCount++;
        
        // Check memory pressure
        if (performanceMetrics.memory) {
          const ramUtilization = performanceMetrics.memory.usedRAM / performanceMetrics.memory.totalRAM;
          memoryPressure = ramUtilization > 0.85;
        }
        
      } catch (error) {
        console.warn('Performance metrics update failed:', error);
      }
    }, 1000);
  }

  async function runAIAnalysis() {
    if (!sampleDocument.trim()) return;
    
    try {
      console.log('🧠 Running Legal-BERT analysis...');
      analysisResult = await legalBERTAnalyzer.analyzeDocument(
        `demo_${Date.now()}`,
        sampleDocument,
        {
          priority: 10,
          useCache: true,
          realTimeUpdates: true,
          includePrecedents: true
        }
      );
      
      console.log('✅ Analysis complete:', analysisResult);
    } catch (error) {
      console.error('❌ AI analysis failed:', error);
    }
  }

  async function testMemoryOperations() {
    try {
      console.log('💾 Testing NES memory operations...');
      
      // Allocate test documents in different memory regions
      const testData = new ArrayBuffer(1024);
      const testDocs = [
        {
          id: 'test_contract_1',
          type: 'contract' as const,
          priority: 200,
          size: 1024,
          confidenceLevel: 0.9,
          riskLevel: 'high' as const,
          compressed: false,
          metadata: { caseId: 'demo_case_1' }
        },
        {
          id: 'test_evidence_1', 
          type: 'evidence' as const,
          priority: 180,
          size: 1024,
          confidenceLevel: 0.85,
          riskLevel: 'medium' as const,
          compressed: false,
          metadata: { caseId: 'demo_case_1' }
        }
      ];
      
      for (const doc of testDocs) {
        await nesMemory.allocateDocument(doc, testData, {
          compress: true,
          preferredBank: doc.type === 'contract' ? 'CHR_ROM' : 'PRG_ROM'
        });
      }
      
      console.log('✅ Memory operations test complete');
    } catch (error) {
      console.error('❌ Memory operations test failed:', error);
    }
  }

  async function testCacheOperations() {
    try {
      console.log('💽 Testing cache operations...');
      
      // Store test documents in cache
      const testDoc = {
        id: 'cache_test_1',
        type: 'brief' as const,
        priority: 150,
        size: 2048,
        confidenceLevel: 0.8,
        riskLevel: 'medium' as const,
        lastAccessed: Date.now(),
        compressed: false,
        metadata: {
          caseId: 'demo_case_cache',
          jurisdiction: 'federal'
        }
      };
      
      await lokiRedisCache.storeDocument(testDoc);
      
      // Retrieve document
      const retrieved = await lokiRedisCache.getDocument('cache_test_1');
      console.log('Retrieved from cache:', retrieved);
      
      // Test search
      const searchResults = await lokiRedisCache.searchDocuments('demo', {
        type: ['brief'],
        confidenceMin: 0.7
      });
      console.log('Search results:', searchResults);
      
      console.log('✅ Cache operations test complete');
    } catch (error) {
      console.error('❌ Cache operations test failed:', error);
    }
  }

  async function testGPUAnimations() {
    try {
      console.log('🎨 Testing GPU animations...');
      
      await gpuAnimations.createAnimation('demo_glow', {
        type: 'legal_confidence_glow',
        duration: 2000,
        legalContext: {
          confidence: 0.85,
          riskLevel: 'medium'
        },
        priority: 5
      });
      
      await gpuAnimations.startAnimation('demo_glow');
      
      setTimeout(async () => {
        await gpuAnimations.stopAnimation('demo_glow');
      }, 3000);
      
      console.log('✅ GPU animations test complete');
    } catch (error) {
      console.error('❌ GPU animations test failed:', error);
    }
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function formatPercent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }

  // Button configurations for different demos
  const getDemoButtonProps = (demoId: string): UnifiedButtonProps => {
    const base = {
      variant: 'primary' as const,
      size: 'md' as const,
      gpuEffects: true,
      glowIntensity: 0.6
    };
    
    switch (demoId) {
      case 'ai_analysis':
        return {
          ...base,
          variant: 'legal',
          legalContext: {
            confidence: 0.9,
            caseType: 'brief',
            aiSuggested: true,
            riskLevel: 'medium'
          }
        };
      case 'memory_architecture':
        return {
          ...base,
          variant: 'evidence',
          nesStyle: true,
          pixelated: true
        };
      case 'webgpu_streaming':
        return {
          ...base,
          variant: 'case',
          gpuEffects: true,
          glowIntensity: 1.0
        };
      default:
        return base;
    }
  };
</script>

<div class=\"min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6\">
  <div class=\"max-w-7xl mx-auto\">
    <!-- Header -->
    <header class=\"mb-8 text-center\">
      <h1 class=\"text-4xl font-bold text-gray-900 mb-2\">
        Phase 14 Complete Demo System
      </h1>
      <p class=\"text-lg text-gray-600 mb-4\">
        Unified UI Kit • WebGPU Streaming • GPU Animations • NES Memory • Loki+Redis Cache • Legal-BERT AI
      </p>
      
      <!-- System Status -->
      {#if isInitialized}
        <div class=\"flex items-center justify-center gap-4 text-sm\">
          <span class=\"flex items-center gap-2 text-green-600\">
            <div class=\"w-2 h-2 bg-green-500 rounded-full animate-pulse\"></div>
            System Online
          </span>
          <span class=\"text-gray-500\">FPS: {fps}</span>
          <span class=\"text-gray-500\">Frame: {frameCount}</span>
          {#if memoryPressure}
            <span class=\"flex items-center gap-2 text-amber-600\">
              <div class=\"w-2 h-2 bg-amber-500 rounded-full animate-pulse\"></div>
              Memory Pressure
            </span>
          {/if}
        </div>
      {:else}
        <div class=\"flex items-center justify-center gap-2 text-amber-600\">
          <div class=\"animate-spin h-4 w-4 border-2 border-amber-600 border-t-transparent rounded-full\"></div>
          Initializing Systems...
        </div>
      {/if}
    </header>

    <!-- Demo Navigation -->
    <nav class=\"mb-8\">
      <div class=\"flex flex-wrap justify-center gap-2\">
        {#each demoScenarios as scenario}
          <UnifiedButton
            {...getDemoButtonProps(scenario.id)}
            variant={currentDemo === scenario.id ? 'primary' : 'secondary'}
            on:on:click={() => currentDemo = scenario.id}
          >
            {scenario.title}
          </UnifiedButton>
        {/each}
      </div>
    </nav>

    <!-- Main Demo Content -->
    <main class=\"grid grid-cols-1 lg:grid-cols-3 gap-6\">
      <!-- Primary Demo Panel -->
      <div class=\"lg:col-span-2\">
        <div class=\"bg-white rounded-xl shadow-lg p-6\">
          {#if currentDemo === 'overview'}
            <div in:fade={{ duration: 300 }}>
              <h2 class=\"text-2xl font-bold mb-4\">Phase 14 System Architecture</h2>
              <div class=\"grid grid-cols-1 md:grid-cols-2 gap-4 mb-6\">
                <!-- Component Status Grid -->
                <div class=\"space-y-3\">
                  <div class=\"flex items-center justify-between p-3 bg-green-50 rounded-lg\">
                    <span class=\"font-medium\">Unified UI Kit</span>
                    <span class=\"text-green-600\">✓ Active</span>
                  </div>
                  <div class=\"flex items-center justify-between p-3 bg-green-50 rounded-lg\">
                    <span class=\"font-medium\">WebGPU Streaming</span>
                    <span class=\"text-green-600\">✓ Active</span>
                  </div>
                  <div class=\"flex items-center justify-between p-3 bg-green-50 rounded-lg\">
                    <span class=\"font-medium\">GPU Animations</span>
                    <span class=\"text-green-600\">✓ Active</span>
                  </div>
                </div>
                <div class=\"space-y-3\">
                  <div class=\"flex items-center justify-between p-3 bg-green-50 rounded-lg\">
                    <span class=\"font-medium\">NES Memory</span>
                    <span class=\"text-green-600\">✓ Active</span>
                  </div>
                  <div class=\"flex items-center justify-between p-3 bg-green-50 rounded-lg\">
                    <span class=\"font-medium\">Loki+Redis Cache</span>
                    <span class=\"text-green-600\">✓ Active</span>
                  </div>
                  <div class=\"flex items-center justify-between p-3 bg-green-50 rounded-lg\">
                    <span class=\"font-medium\">Legal-BERT AI</span>
                    <span class=\"text-green-600\">✓ Active</span>
                  </div>
                </div>
              </div>
              
              <div class=\"flex flex-wrap gap-3\">
                <UnifiedButton on:on:click={testMemoryOperations} variant=\"evidence\">
                  Test Memory Operations
                </UnifiedButton>
                <UnifiedButton on:on:click={testCacheOperations} variant=\"case\">
                  Test Cache Operations
                </UnifiedButton>
                <UnifiedButton on:on:click={testGPUAnimations} variant=\"legal\">
                  Test GPU Animations
                </UnifiedButton>
              </div>
            </div>
          
          {:else if currentDemo === 'ui_components'}
            <div in:fade={{ duration: 300 }}>
              <h2 class=\"text-2xl font-bold mb-4\">Unified UI Components</h2>
              <p class=\"text-gray-600 mb-6\">
                Integration of bits-ui v2 + Melt Svelte 5 + UnoCSS with legal AI context awareness
              </p>
              
              <div class=\"space-y-6\">
                <!-- Button Variants -->
                <div>
                  <h3 class=\"text-lg font-semibold mb-3\">Button Variants</h3>
                  <div class=\"flex flex-wrap gap-3\">
                    <UnifiedButton variant=\"primary\">Primary</UnifiedButton>
                    <UnifiedButton variant=\"secondary\">Secondary</UnifiedButton>
                    <UnifiedButton variant=\"legal\" legalContext={{
                      confidence: 0.95,
                      caseType: 'contract',
                      aiSuggested: true,
                      riskLevel: 'low'
                    }}>Legal (AI: 95%)</UnifiedButton>
                    <UnifiedButton variant=\"evidence\" legalContext={{
                      confidence: 0.7,
                      caseType: 'evidence',
                      riskLevel: 'high'
                    }}>Evidence (Risk: High)</UnifiedButton>
                    <UnifiedButton variant=\"case\" nesStyle pixelated>NES Style</UnifiedButton>
                    <UnifiedButton variant=\"ghost\" loading>Loading...</UnifiedButton>
                  </div>
                </div>
                
                <!-- Size Variants -->
                <div>
                  <h3 class=\"text-lg font-semibold mb-3\">Size Variants</h3>
                  <div class=\"flex items-end gap-3\">
                    <UnifiedButton size=\"sm\" variant=\"legal\">Small</UnifiedButton>
                    <UnifiedButton size=\"md\" variant=\"legal\">Medium</UnifiedButton>
                    <UnifiedButton size=\"lg\" variant=\"legal\">Large</UnifiedButton>
                    <UnifiedButton size=\"xl\" variant=\"legal\">Extra Large</UnifiedButton>
                  </div>
                </div>
                
                <!-- Dialog Demo -->
                <div>
                  <h3 class=\"text-lg font-semibold mb-3\">Modal Dialog</h3>
                  <UnifiedButton on:on:click={() => showDialog = true} variant=\"case\">
                    Open Legal Analysis Dialog
                  </UnifiedButton>
                </div>
              </div>
            </div>
          
          {:else if currentDemo === 'ai_analysis'}
            <div in:fade={{ duration: 300 }}>
              <h2 class=\"text-2xl font-bold mb-4\">Legal-BERT Semantic Analysis</h2>
              <p class=\"text-gray-600 mb-6\">
                Real-time legal document analysis using transformer-based models
              </p>
              
              <!-- Document Input -->
              <div class=\"mb-6\">
                <label class=\"block text-sm font-medium mb-2\">Legal Document</label>
                <div class=\"mb-3\">
                  <select 
                    class=\"block w-full px-3 py-2 border border-gray-300 rounded-md\"
                    change={(e) => sampleDocument = sampleDocuments[parseInt(e.currentTarget.value)]?.content || ''}
                  >
                    {#each sampleDocuments as doc, index}
                      <option value={index}>{doc.title}</option>
                    {/each}
                  </select>
                </div>
                <textarea
                  bind:value={sampleDocument}
                  rows=\"6\"
                  class=\"block w-full px-3 py-2 border border-gray-300 rounded-md resize-none\"
                  placeholder=\"Enter legal document text for analysis...\"
                ></textarea>
              </div>
              
              <div class=\"mb-6\">
                <UnifiedButton 
                  on:on:click={runAIAnalysis} 
                  variant=\"legal\"
                  disabled={!sampleDocument.trim()}
                  legalContext={{
                    confidence: 0.9,
                    caseType: 'brief',
                    aiSuggested: true,
                    riskLevel: 'medium'
                  }}
                >
                  Analyze with Legal-BERT
                </UnifiedButton>
              </div>
              
              <!-- Analysis Results -->
              {#if analysisResult}
                <div class=\"space-y-4\" in:fly={{ y: 20, duration: 500 }}>
                  <!-- Classification -->
                  <div class=\"p-4 bg-blue-50 rounded-lg\">
                    <h4 class=\"font-semibold text-blue-900 mb-2\">Document Classification</h4>
                    <div class=\"flex items-center gap-3\">
                      <span class=\"px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-medium\">
                        {analysisResult.classification.category}
                      </span>
                      <span class=\"text-blue-700\">
                        Confidence: {Math.round(analysisResult.classification.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                  
                  <!-- Risk Assessment -->
                  <div class=\"p-4 bg-red-50 rounded-lg\">
                    <h4 class=\"font-semibold text-red-900 mb-2\">Risk Assessment</h4>
                    <div class=\"flex items-center gap-3 mb-2\">
                      <span class=\"px-2 py-1 bg-red-100 text-red-800 rounded text-sm font-medium capitalize\">
                        {analysisResult.riskAssessment.overallRisk}
                      </span>
                      <span class=\"text-red-700\">
                        Score: {Math.round(analysisResult.riskAssessment.riskScore)}/100
                      </span>
                    </div>
                    {#if analysisResult.riskAssessment.riskFactors.length > 0}
                      <div class=\"mt-3\">
                        <p class=\"text-sm text-red-700 mb-2\">Risk Factors:</p>
                        <ul class=\"text-sm space-y-1\">
                          {#each analysisResult.riskAssessment.riskFactors.slice(0, 3) as factor}
                            <li class=\"text-red-600\">
                              • {factor.factor.replace(/_/g, ' ')} 
                              <span class=\"text-red-500\">({factor.severity})</span>
                            </li>
                          {/each}
                        </ul>
                      </div>
                    {/if}
                  </div>
                  
                  <!-- Entities -->
                  {#if analysisResult.entities.length > 0}
                    <div class=\"p-4 bg-green-50 rounded-lg\">
                      <h4 class=\"font-semibold text-green-900 mb-2\">Legal Entities</h4>
                      <div class=\"flex flex-wrap gap-2\">
                        {#each analysisResult.entities.slice(0, 6) as entity}
                          <span class=\"px-2 py-1 bg-green-100 text-green-800 rounded text-xs\">
                            {entity.text} ({entity.label})
                          </span>
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}
            </div>
          
          {:else if currentDemo === 'performance'}
            <div in:fade={{ duration: 300 }}>
              <h2 class=\"text-2xl font-bold mb-4\">Performance Dashboard</h2>
              <p class=\"text-gray-600 mb-6\">
                Real-time system performance monitoring and metrics
              </p>
              
              <!-- Performance Grid -->
              <div class=\"grid grid-cols-1 md:grid-cols-2 gap-4\">
                <!-- Memory Stats -->
                {#if performanceMetrics.memory}
                  <div class=\"p-4 bg-purple-50 rounded-lg\">
                    <h4 class=\"font-semibold text-purple-900 mb-3\">NES Memory Architecture</h4>
                    <div class=\"space-y-2 text-sm\">
                      <div class=\"flex justify-between\">
                        <span>RAM Usage:</span>
                        <span>{formatBytes(performanceMetrics.memory.usedRAM)} / {formatBytes(performanceMetrics.memory.totalRAM)}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>CHR-ROM:</span>
                        <span>{formatBytes(performanceMetrics.memory.usedCHR)} / {formatBytes(performanceMetrics.memory.totalCHR)}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>PRG-ROM:</span>
                        <span>{formatBytes(performanceMetrics.memory.usedPRG)} / {formatBytes(performanceMetrics.memory.totalPRG)}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Documents:</span>
                        <span>{performanceMetrics.memory.documentCount}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Bank Switches:</span>
                        <span>{performanceMetrics.memory.bankSwitches}</span>
                      </div>
                    </div>
                  </div>
                {/if}
                
                <!-- Cache Stats -->
                {#if performanceMetrics.cache}
                  <div class=\"p-4 bg-cyan-50 rounded-lg\">
                    <h4 class=\"font-semibold text-cyan-900 mb-3\">Cache Performance</h4>
                    <div class=\"space-y-2 text-sm\">
                      <div class=\"flex justify-between\">
                        <span>Hit Ratio:</span>
                        <span>{formatPercent(performanceMetrics.cache.overall.hitRatio)}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Loki Documents:</span>
                        <span>{performanceMetrics.cache.loki.documents}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Redis Connected:</span>
                        <span class={performanceMetrics.cache.redis.connected ? 'text-green-600' : 'text-red-600'}>
                          {performanceMetrics.cache.redis.connected ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Total Documents:</span>
                        <span>{performanceMetrics.cache.overall.totalDocuments}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Avg Response:</span>
                        <span>{Math.round(performanceMetrics.cache.overall.avgResponseTime)}ms</span>
                      </div>
                    </div>
                  </div>
                {/if}
                
                <!-- AI Stats -->
                {#if performanceMetrics.ai}
                  <div class=\"p-4 bg-amber-50 rounded-lg\">
                    <h4 class=\"font-semibold text-amber-900 mb-3\">Legal-BERT AI</h4>
                    <div class=\"space-y-2 text-sm\">
                      <div class=\"flex justify-between\">
                        <span>Documents Analyzed:</span>
                        <span>{performanceMetrics.ai.documentsAnalyzed}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Avg Processing Time:</span>
                        <span>{Math.round(performanceMetrics.ai.averageProcessingTime)}ms</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Cache Hit Ratio:</span>
                        <span>{formatPercent(performanceMetrics.ai.cacheHitRatio)}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Entities Extracted:</span>
                        <span>{performanceMetrics.ai.entitiesExtracted}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Risks Identified:</span>
                        <span>{performanceMetrics.ai.risksIdentified}</span>
                      </div>
                    </div>
                  </div>
                {/if}
                
                <!-- GPU Stats -->
                {#if performanceMetrics.gpu}
                  <div class=\"p-4 bg-indigo-50 rounded-lg\">
                    <h4 class=\"font-semibold text-indigo-900 mb-3\">GPU Performance</h4>
                    <div class=\"space-y-2 text-sm\">
                      <div class=\"flex justify-between\">
                        <span>Active Animations:</span>
                        <span>{performanceMetrics.gpu.activeAnimations || 0}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Memory Used:</span>
                        <span>{formatBytes(performanceMetrics.gpu.memoryUsed || 0)}</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Frame Rate:</span>
                        <span>{fps} FPS</span>
                      </div>
                      <div class=\"flex justify-between\">
                        <span>Render Time:</span>
                        <span>{performanceMetrics.gpu.avgRenderTime || 0}ms</span>
                      </div>
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          
          {:else}
            <div in:fade={{ duration: 300 }}>
              <h2 class=\"text-2xl font-bold mb-4\">
                {demoScenarios.find(s => s.id === currentDemo)?.title || 'Demo'}
              </h2>
              <p class=\"text-gray-600 mb-6\">
                {demoScenarios.find(s => s.id === currentDemo)?.description || 'Demo content'}
              </p>
              <div class=\"text-center py-12 text-gray-500\">
                Demo content for {currentDemo} is under construction.
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Metrics Sidebar -->
      <div class=\"space-y-6\">
        <!-- System Status -->
        <div class=\"bg-white rounded-xl shadow-lg p-4\">
          <h3 class=\"font-semibold text-gray-900 mb-3\">System Status</h3>
          <div class=\"space-y-2 text-sm\">
            <div class=\"flex items-center justify-between\">
              <span>Initialization:</span>
              <span class={isInitialized ? 'text-green-600' : 'text-amber-600'}>
                {isInitialized ? '✓ Ready' : '⏳ Loading'}
              </span>
            </div>
            <div class=\"flex items-center justify-between\">
              <span>Frame Rate:</span>
              <span class={fps > 30 ? 'text-green-600' : fps > 15 ? 'text-amber-600' : 'text-red-600'}>
                {fps} FPS
              </span>
            </div>
            <div class=\"flex items-center justify-between\">
              <span>Memory Pressure:</span>
              <span class={memoryPressure ? 'text-red-600' : 'text-green-600'}>
                {memoryPressure ? '⚠ High' : '✓ Normal'}
              </span>
            </div>
          </div>
        </div>

        <!-- Quick Actions -->
        <div class=\"bg-white rounded-xl shadow-lg p-4\">
          <h3 class=\"font-semibold text-gray-900 mb-3\">Quick Actions</h3>
          <div class=\"space-y-2\">
            <UnifiedButton 
              on:on:click={runAIAnalysis} 
              variant=\"legal\" 
              size=\"sm\"
              disabled={!sampleDocument.trim()}
            >
              Run AI Analysis
            </UnifiedButton>
            <UnifiedButton 
              on:on:click={testMemoryOperations} 
              variant=\"evidence\" 
              size=\"sm\"
            >
              Test Memory
            </UnifiedButton>
            <UnifiedButton 
              on:on:click={testGPUAnimations} 
              variant=\"case\" 
              size=\"sm\"
            >
              Test GPU
            </UnifiedButton>
          </div>
        </div>

        <!-- Component Links -->
        <div class=\"bg-white rounded-xl shadow-lg p-4\">
          <h3 class=\"font-semibold text-gray-900 mb-3\">Phase 14 Components</h3>
          <div class=\"space-y-1 text-xs\">
            <div class=\"text-green-600\">✓ Unified UI Kit</div>
            <div class=\"text-green-600\">✓ WebGPU Streaming</div>
            <div class=\"text-green-600\">✓ GPU Animations</div>
            <div class=\"text-green-600\">✓ NES Memory</div>
            <div class=\"text-green-600\">✓ Loki+Redis Cache</div>
            <div class=\"text-green-600\">✓ Legal-BERT AI</div>
            <div class=\"text-green-600\">✓ Performance Monitor</div>
          </div>
        </div>
      </div>
    </main>
  </div>
</div>

<!-- Legal Analysis Dialog -->
<UnifiedDialog
  bind:open={showDialog}
  size=\"lg\"
  variant=\"legal\"
  webgpuEffects={true}
  legalContext={{
    caseId: 'demo_case_1',
    documentType: 'contract',
    aiAnalysis: {
      riskLevel: 'medium',
      confidence: 0.85,
      suggestions: [
        'Review indemnification clauses for liability limits',
        'Consider adding force majeure provisions',
        'Verify termination procedures are clearly defined'
      ]
    }
  }}
>
  {#snippet title()}
    Legal Document Analysis Results
  {/snippet}
  
  {#snippet content()}
    <div class=\"space-y-4\">
      {#if analysisResult}
        <div class=\"grid grid-cols-2 gap-4\">
          <div>
            <h4 class=\"font-semibold mb-2\">Classification</h4>
            <p class=\"text-sm text-gray-600 capitalize\">{analysisResult.classification.category}</p>
            <p class=\"text-xs text-gray-500\">Confidence: {Math.round(analysisResult.classification.confidence * 100)}%</p>
          </div>
          <div>
            <h4 class=\"font-semibold mb-2\">Risk Level</h4>
            <p class=\"text-sm text-gray-600 capitalize\">{analysisResult.riskAssessment.overallRisk}</p>
            <p class=\"text-xs text-gray-500\">Score: {Math.round(analysisResult.riskAssessment.riskScore)}/100</p>
          </div>
        </div>
        
        <div>
          <h4 class=\"font-semibold mb-2\">Key Metrics</h4>
          <ul class=\"text-sm space-y-1\">
            <li>Entities Extracted: {analysisResult.entities.length}</li>
            <li>Risk Factors: {analysisResult.riskAssessment.riskFactors.length}</li>
            <li>Processing Time: {analysisResult.processingTime}ms</li>
            <li>Reading Level: {analysisResult.complexity.readingLevel.toFixed(1)}</li>
          </ul>
        </div>
      {:else}
        <p class=\"text-center text-gray-500 py-8\">
          Run an AI analysis to see detailed results here.
        </p>
      {/if}
    </div>
  {/snippet}
  
  {#snippet footer()}
    <UnifiedButton 
      on:on:click={() => showDialog = false} 
      variant=\"secondary\"
    >
      Close
    </UnifiedButton>
  {/snippet}
</UnifiedDialog>

<style>
  /* Custom animations for demo system */
  @keyframes pulse-glow {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }
  
  .animate-pulse-glow {
    animation: pulse-glow 2s ease-in-out infinite;
  }
  
  /* Performance optimizations */
  * {
    will-change: auto;
  }
  
  .gpu-accelerated {
    transform: translateZ(0);
    backface-visibility: hidden;
  }
</style>